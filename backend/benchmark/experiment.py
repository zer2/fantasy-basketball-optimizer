# benchmark/experiment.py
"""Orchestrate the benchmark grid: field x format x temperature, with common random numbers."""
import numpy as np

from backend.benchmark.agents import GScoreAgent, HScoreAgent
from backend.benchmark.config import LeagueConfig, ExperimentConfig
from backend.benchmark.mcts import MCTSAgent, _top_k_eligible
from backend.benchmark.opponent_model import weighted_softmax_pick
from backend.benchmark.draft import run_draft


class _FastFieldAgent:
    """Speed wrapper around a GScoreAgent/HScoreAgent.

    Behaviorally identical to the wrapped agent: it pre-slices the candidate pool
    to the first ``top_k`` position-eligible players in ranking order (early-stopping
    the eligibility LP) and hands that prefix to ``weighted_softmax_pick``. Because
    ``weighted_softmax_pick`` itself restricts to ranking order, re-checks eligibility,
    and takes ``[:top_k]``, it sees the identical candidate set / scores / rng draw as
    when passed the full available list — only far fewer eligibility solves. This is the
    same prefix trick the MCTS rollout uses, and it is what keeps a full field draft to
    seconds instead of ~1.5 minutes.
    """

    def __init__(self, base):
        self.ranking = base.ranking
        self.positions = base.positions
        self.temperature = base.temperature
        self.top_k = base.top_k

    def make_pick(self, player_assignments, seat, rng):
        taken = {p for v in player_assignments.values() for p in v if p == p}
        cand = _top_k_eligible(self.ranking, taken, self.positions,
                               player_assignments[seat], self.top_k)
        return weighted_softmax_pick(self.ranking, cand, self.temperature, rng,
                                     self.positions, player_assignments[seat], self.top_k)


def _make_field(info, cfg, field, temperature, exp_cfg):
    if field == 'gscore':
        return lambda: _FastFieldAgent(
            GScoreAgent(info, temperature=temperature, top_k=exp_cfg.mcts_top_k))
    elif field == 'hscore':
        return lambda: _FastFieldAgent(
            HScoreAgent(info, cfg, temperature=temperature, top_k=exp_cfg.mcts_top_k))
    raise ValueError(field)


def run_matched_draft(info, cfg, exp_cfg, field, fmt, temperature, hero_seat, seed):
    field_factory = _make_field(info, cfg, field, temperature, exp_cfg)

    def build(hero_agent):
        agents = {s: field_factory() for s in range(cfg.n_drafters)}
        agents[hero_seat] = hero_agent
        return agents

    # H-hero is deterministic (temperature 0); wrap for the same early-stop speedup.
    hero_h_agent = _FastFieldAgent(
        HScoreAgent(info, cfg, temperature=0.0, top_k=exp_cfg.mcts_top_k))
    hero_m_agent = MCTSAgent(info, cfg, temperature=temperature,
                             n_simulations=exp_cfg.mcts_simulations,
                             top_k=exp_cfg.mcts_top_k, c_puct=exp_cfg.c_puct)

    # Common random numbers: each seat gets its OWN persistent generator seeded from
    # (seed, seat). A single shared rng desyncs the field because the two heros (H vs
    # MCTS) consume different numbers of draws, shifting every downstream seat's stream.
    # Per-seat streams isolate the hero's draws so non-hero seats see identical randomness
    # across both runs.
    def rng_for_seat(s):
        return np.random.default_rng([seed, s])

    res_h = run_draft(build(hero_h_agent), cfg.n_drafters, cfg.n_starters, rng_for_seat)
    res_m = run_draft(build(hero_m_agent), cfg.n_drafters, cfg.n_starters, rng_for_seat)
    return res_h, res_m


# --- Task 12: full grid + result aggregation + CLI ---
import json
from dataclasses import replace

from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.evaluate import evaluate_rosters

def _aggregate(hero_ec, hero_mc):
    arr_ec = np.array(hero_ec); arr_mc = np.array(hero_mc)
    def stat(a):
        return {'mean': float(a.mean()),
                'ci': float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0}
    return {'EC': stat(arr_ec)['mean'], 'EC_ci': stat(arr_ec)['ci'],
            'MC': stat(arr_mc)['mean'], 'MC_ci': stat(arr_mc)['ci']}

def run_experiment(exp_cfg, cfg, fixture_path):
    averages, gamelogs = load_fixture(fixture_path)
    results = {}
    for fmt in exp_cfg.formats:
        cfg_fmt = replace(cfg, scoring_format=fmt)
        info = bootstrap_session(averages, cfg_fmt)
        for field in exp_cfg.fields:
            for T in exp_cfg.temperatures:
                h_ec, h_mc, m_ec, m_mc = [], [], [], []
                for d in range(exp_cfg.n_drafts):
                    hero_seat = d % cfg.n_drafters
                    seed = exp_cfg.seed + d
                    res_h, res_m = run_matched_draft(info, cfg_fmt, exp_cfg, field, fmt, T, hero_seat, seed)
                    eval_rng = np.random.default_rng([exp_cfg.seed, d])
                    ev_h = evaluate_rosters(res_h, gamelogs, cfg_fmt, exp_cfg, eval_rng)
                    eval_rng = np.random.default_rng([exp_cfg.seed, d])  # same season draws
                    ev_m = evaluate_rosters(res_m, gamelogs, cfg_fmt, exp_cfg, eval_rng)
                    key = 'EC' if fmt.endswith('Each Category') else 'MC'
                    h_ec.append(ev_h[hero_seat]['EC']); h_mc.append(ev_h[hero_seat]['MC'])
                    m_ec.append(ev_m[hero_seat]['EC']); m_mc.append(ev_m[hero_seat]['MC'])
                results[(field, fmt, T)] = {
                    'hscore_hero': _aggregate(h_ec, h_mc),
                    'mcts_hero': _aggregate(m_ec, m_mc),
                    'delta_EC': float(np.mean(m_ec) - np.mean(h_ec)),
                    'delta_MC': float(np.mean(m_mc) - np.mean(h_mc)),
                }
    return results

def main(timestamp='manual'):
    results = run_experiment(ExperimentConfig(), LeagueConfig(),
                             'backend/benchmark/fixtures/nba_2025-26.parquet')
    serializable = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in results.items()}
    import os
    os.makedirs('benchmark/results', exist_ok=True)
    with open(f'benchmark/results/{timestamp}.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    for k, v in serializable.items():
        print(k, 'ΔEC=%+.3f ΔMC=%+.3f' % (v['delta_EC'], v['delta_MC']))
    return results

if __name__ == '__main__':
    main()
