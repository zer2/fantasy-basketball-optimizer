# Draft Strategy Benchmark

Measures whether MCTS lookahead beats one-ply H-score at drafting, scored by an
independent Monte-Carlo of the real 2025-26 NBA fantasy season.

**Additive only:** nothing under `src/` or `app.py` is modified; the engine is
imported read-only. `benchmark/tests/test_no_src_edits.py` enforces this.

## Run
```bash
# rebuild the fixture (network, one-time):
SPORT=NBA .venv/bin/python -c "from benchmark.data import pull_and_snapshot; pull_and_snapshot('2025-26','benchmark/fixtures/nba_2025-26.parquet')"
# full benchmark:
SPORT=NBA .venv/bin/python -m benchmark.experiment
# tests:
SPORT=NBA .venv/bin/python -m pytest benchmark/tests/ -v
```

## Reading results
Each grid cell reports hero EC/MC win-rate for H-score vs. MCTS and the deltas
(`ΔEC`, `ΔMC`) as a function of field type and opponent temperature `T`.
Positive delta = MCTS beats H-score. The headline is the delta-vs-`T` curve.
