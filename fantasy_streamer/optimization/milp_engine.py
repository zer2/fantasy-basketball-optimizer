"""Rolling-horizon MILP solver engine using Google OR-Tools or PuLP (HiGHS)."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from fantasy_streamer.config import LeagueConfig
from fantasy_streamer.models.projections import Player
from fantasy_streamer.models.schedule import MatchupHorizon, ScheduleMatrix
from fantasy_streamer.optimization.constraints import (
    ConstraintManager,
    OptimizationProblemData,
)


@dataclass
class StreamingAction:
    """Represents a recommended transaction on a specific day."""

    day_idx: int
    date_str: str
    action_type: str  # "ADD", "DROP", or "HOLD"
    player_id: int
    player_name: str
    player_team: str
    fppg: float
    reason: str


@dataclass
class DailyLineupPlan:
    """Recommended roster and lineup allocation for a specific day."""

    day_idx: int
    date_str: str
    week_num: int
    starters: Dict[str, str]  # slot -> player_name
    bench: List[str]  # list of player_names on bench
    total_projected_fp: float
    active_games_count: int


@dataclass
class StreamingOptimizationResult:
    """Full output of the rolling-horizon MILP optimization."""

    status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE"
    solve_time_ms: float = 0.0  # Pure solver engine solve time (e.g. CBC / SCIP / HiGHS / CP-SAT)
    total_time_ms: float = 0.0  # Total end-to-end time including model build & extraction
    total_projected_fp: float = 0.0
    net_objective_value: float = 0.0
    total_drop_penalties: float = 0.0
    total_adds_week1: int = 0
    total_adds_week2: int = 0
    actions: List[StreamingAction] = field(default_factory=list)
    daily_plans: List[DailyLineupPlan] = field(default_factory=list)
    retained_bench_holds: List[str] = field(default_factory=list)
    dropped_bench_holds: List[str] = field(default_factory=list)


class MILPStreamerEngine:
    """Core solver engine orchestrating rolling-horizon optimization."""

    def __init__(self, backend: str = "ortools"):
        """Initialize engine with preferred backend ('ortools' or 'pulp')."""
        self.backend = backend.lower()
        if self.backend == "ortools" and not ORTOOLS_AVAILABLE:
            if PULP_AVAILABLE:
                self.backend = "pulp"
            else:
                raise RuntimeError("Neither ortools nor pulp is installed.")
        elif self.backend == "pulp" and not PULP_AVAILABLE:
            if ORTOOLS_AVAILABLE:
                self.backend = "ortools"
            else:
                raise RuntimeError("Neither pulp nor ortools is installed.")

    def solve(self, data: OptimizationProblemData) -> StreamingOptimizationResult:
        """Execute MILP solve using the configured backend."""
        start_time = time.perf_counter()
        if self.backend == "ortools":
            result = self._solve_ortools(data)
        else:
            result = self._solve_pulp(data)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        result.total_time_ms = total_time_ms
        return result

    def _solve_ortools(
        self, data: OptimizationProblemData
    ) -> StreamingOptimizationResult:
        """Solve using Google OR-Tools CBC/SCIP solver."""
        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("Could not create OR-Tools solver backend (CBC or SCIP).")

        # Set time limit to 5000ms (we expect <50ms for pruned pools)
        solver.SetTimeLimit(5000)

        cm = ConstraintManager(data)
        horizon_days = data.horizon.total_days
        last_day = horizon_days - 1

        all_players = data.all_players
        droppable_roster = data.droppable_roster
        non_droppable = data.non_droppable_roster
        free_agents = data.candidate_free_agents
        fa_ids = {p.player_id for p in free_agents}

        # -------------------------------------------------------------
        # Decision Variables
        # -------------------------------------------------------------
        # r[p, d]: binary roster ownership
        r_var: Dict[Tuple[int, int], pywraplp.Variable] = {}
        for p in all_players:
            for d in range(horizon_days):
                r_var[(p.player_id, d)] = solver.BoolVar(f"r_{p.player_id}_{d}")

        # a[p, d]: binary acquisition indicator for free agents
        a_var: Dict[Tuple[int, int], pywraplp.Variable] = {}
        for p in free_agents:
            for d in range(horizon_days):
                a_var[(p.player_id, d)] = solver.BoolVar(f"a_{p.player_id}_{d}")

        # x[p, l, d]: binary starter indicator
        # Only create if player has a game on day d AND is eligible for slot l
        x_var: Dict[Tuple[int, str, int], pywraplp.Variable] = {}
        slot_day_vars: Dict[Tuple[str, int], List[pywraplp.Variable]] = {
            (s, d): [] for s in data.config.active_slots for d in range(horizon_days)
        }
        player_day_vars: Dict[Tuple[int, int], List[pywraplp.Variable]] = {
            (p.player_id, d): [] for p in all_players for d in range(horizon_days)
        }

        for p in all_players:
            pid = p.player_id
            for d in range(horizon_days):
                if cm.has_game(pid, d):
                    for slot in cm.eligible_slots_map[pid]:
                        var = solver.BoolVar(f"x_{pid}_{slot}_{d}")
                        x_var[(pid, slot, d)] = var
                        slot_day_vars[(slot, d)].append(var)
                        player_day_vars[(pid, d)].append(var)

        # -------------------------------------------------------------
        # Constraints
        # -------------------------------------------------------------
        # 1. Non-droppable core players must remain on roster for entire horizon
        for p in non_droppable:
            for d in range(horizon_days):
                solver.Add(r_var[(p.player_id, d)] == 1)

        # 2. Droppable roster holds: Monotonic hold/drop rule (r_p,d <= r_p,d-1)
        for p in droppable_roster:
            for d in range(1, horizon_days):
                solver.Add(r_var[(p.player_id, d)] <= r_var[(p.player_id, d - 1)])

        # 3. Free agent acquisition logic: a_p,d >= r_p,d - r_p,d-1
        for p in free_agents:
            pid = p.player_id
            solver.Add(a_var[(pid, 0)] >= r_var[(pid, 0)])
            for d in range(1, horizon_days):
                solver.Add(a_var[(pid, d)] >= r_var[(pid, d)] - r_var[(pid, d - 1)])

        # 4. Total roster capacity per day: sum_p r_p,d <= config.roster_size (13)
        for d in range(horizon_days):
            solver.Add(
                solver.Sum(r_var[(p.player_id, d)] for p in all_players)
                <= data.config.roster_size
            )

        # 4b. Drops conservation: Initial droppable roster holds can only be dropped if
        # accompanied by cumulative free agent acquisitions on or before day d.
        # sum_{p in droppable} (1 - r_{p,d}) <= sum_{p in FA, t <= d} a_{p,t}
        if free_agents and droppable_roster:
            for d in range(horizon_days):
                cum_adds = [
                    a_var[(p.player_id, t)]
                    for p in free_agents
                    for t in range(d + 1)
                ]
                solver.Add(
                    solver.Sum(1 - r_var[(p.player_id, d)] for p in droppable_roster)
                    <= solver.Sum(cum_adds)
                )
        elif droppable_roster and not free_agents:
            # If no free agents available, droppable roster holds cannot be dropped
            for p in droppable_roster:
                for d in range(horizon_days):
                    solver.Add(r_var[(p.player_id, d)] == 1)

        # 5. Slot capacity: At most 1 player per slot per day
        for d in range(horizon_days):
            for slot in data.config.active_slots:
                vars_in_slot = slot_day_vars[(slot, d)]
                if vars_in_slot:
                    solver.Add(solver.Sum(vars_in_slot) <= 1)

        # 6. Single slot per player per day & must be rostered
        for p in all_players:
            pid = p.player_id
            for d in range(horizon_days):
                p_day_vars = player_day_vars[(pid, d)]
                if p_day_vars:
                    solver.Add(solver.Sum(p_day_vars) <= r_var[(pid, d)])

        # 7. Partitioned weekly acquisition limits
        # Week 1 adds
        w1_add_vars = [
            a_var[(p.player_id, d)]
            for p in free_agents
            for d in data.horizon.week1_days
        ]
        if w1_add_vars:
            solver.Add(solver.Sum(w1_add_vars) <= data.rem_acquisitions_w1)

        # Week 2 adds
        w2_add_vars = [
            a_var[(p.player_id, d)]
            for p in free_agents
            for d in data.horizon.week2_days
        ]
        if w2_add_vars:
            solver.Add(solver.Sum(w2_add_vars) <= data.cap_acquisitions_w2)

        # -------------------------------------------------------------
        # Objective Function
        # -------------------------------------------------------------
        # Maximize: sum (U_w * gamma^d * fppg * x) + sum (c_p * r_p,end) - epsilon * sum(a)
        # Note: - c_p * (1 - r_end) = c_p * r_end - c_p
        objective = solver.Objective()
        objective.SetMaximization()

        for (pid, slot, d), var in x_var.items():
            player = cm.player_map[pid]
            coeff = cm.day_weights[d] * player.fppg
            objective.SetCoefficient(var, coeff)

        for p in droppable_roster:
            # Retention incentive: drop penalty c_p at end of horizon plus infinitesimal
            # hold tie-breaker (1e-5 daily, 1e-4 at horizon end) to guarantee zero-cost
            # roster holds are never dropped spuriously due to degeneracy when c_p == 0.
            for d in range(horizon_days):
                var_d = r_var[(p.player_id, d)]
                objective.SetCoefficient(var_d, objective.GetCoefficient(var_d) + 1e-5)
            var_last = r_var[(p.player_id, last_day)]
            current_coeff = objective.GetCoefficient(var_last)
            objective.SetCoefficient(
                var_last, current_coeff + p.drop_penalty_c_p + 1e-4
            )

        for (pid, d), var in a_var.items():
            objective.SetCoefficient(var, -data.add_friction_epsilon)

        t_solve_0 = time.perf_counter()
        status_code = solver.Solve()
        solver_wall_ms = (time.perf_counter() - t_solve_0) * 1000.0

        if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return StreamingOptimizationResult(
                status="INFEASIBLE",
                solve_time_ms=solver_wall_ms,
                total_time_ms=0.0,
                total_projected_fp=0.0,
                net_objective_value=0.0,
                total_drop_penalties=0.0,
                total_adds_week1=0,
                total_adds_week2=0,
            )

        # -------------------------------------------------------------
        # Solution Extraction
        # -------------------------------------------------------------
        res = self._extract_solution(
            data=data,
            cm=cm,
            status_str="OPTIMAL" if status_code == pywraplp.Solver.OPTIMAL else "FEASIBLE",
            get_r_val=lambda pid, d: r_var[(pid, d)].solution_value() > 0.5,
            get_a_val=lambda pid, d: a_var[(pid, d)].solution_value() > 0.5,
            get_x_val=lambda pid, s, d: (
                x_var[(pid, s, d)].solution_value() > 0.5
                if (pid, s, d) in x_var
                else False
            ),
        )
        res.solve_time_ms = solver_wall_ms
        return res

    def _solve_pulp(
        self, data: OptimizationProblemData
    ) -> StreamingOptimizationResult:
        """Solve using PuLP (CPSAT, HiGHS, or default CBC solver)."""
        prob = pulp.LpProblem("FantasyStreamer_MILP", pulp.LpMaximize)

        cm = ConstraintManager(data)
        horizon_days = data.horizon.total_days
        last_day = horizon_days - 1

        all_players = data.all_players
        droppable_roster = data.droppable_roster
        non_droppable = data.non_droppable_roster
        free_agents = data.candidate_free_agents

        def _make_binary_var(name: str):
            if hasattr(prob, "add_variable"):
                return prob.add_variable(name, cat="Binary")
            return pulp.LpVariable(name, cat="Binary")

        # Variables
        r_var = {}
        for p in all_players:
            for d in range(horizon_days):
                r_var[(p.player_id, d)] = _make_binary_var(f"r_{p.player_id}_{d}")

        a_var = {}
        for p in free_agents:
            for d in range(horizon_days):
                a_var[(p.player_id, d)] = _make_binary_var(f"a_{p.player_id}_{d}")

        x_var = {}
        for p in all_players:
            for d in range(horizon_days):
                if cm.has_game(p.player_id, d):
                    for slot in cm.eligible_slots_map[p.player_id]:
                        x_var[(p.player_id, slot, d)] = _make_binary_var(
                            f"x_{p.player_id}_{slot}_{d}"
                        )

        # Objective terms
        obj_terms = []
        for (pid, slot, d), var in x_var.items():
            player = cm.player_map[pid]
            coeff = cm.day_weights[d] * player.fppg
            obj_terms.append(coeff * var)

        for p in droppable_roster:
            # Retention incentive: drop penalty c_p at end of horizon plus infinitesimal
            # hold tie-breaker (1e-5 daily, 1e-4 at horizon end) to guarantee zero-cost
            # roster holds are never dropped spuriously due to degeneracy when c_p == 0.
            for d in range(horizon_days):
                obj_terms.append(1e-5 * r_var[(p.player_id, d)])
            obj_terms.append(
                (p.drop_penalty_c_p + 1e-4) * r_var[(p.player_id, last_day)]
            )

        for (pid, d), var in a_var.items():
            obj_terms.append(-data.add_friction_epsilon * var)

        prob += pulp.lpSum(obj_terms)

        # Constraints
        for p in non_droppable:
            for d in range(horizon_days):
                prob += r_var[(p.player_id, d)] == 1

        for p in droppable_roster:
            for d in range(1, horizon_days):
                prob += r_var[(p.player_id, d)] <= r_var[(p.player_id, d - 1)]

        for p in free_agents:
            pid = p.player_id
            prob += a_var[(pid, 0)] >= r_var[(pid, 0)]
            for d in range(1, horizon_days):
                prob += a_var[(pid, d)] >= r_var[(pid, d)] - r_var[(pid, d - 1)]

        for d in range(horizon_days):
            prob += (
                pulp.lpSum([r_var[(p.player_id, d)] for p in all_players])
                <= data.config.roster_size
            )

        # Drops conservation: Initial droppable roster holds can only be dropped if
        # accompanied by cumulative free agent acquisitions on or before day d.
        if free_agents and droppable_roster:
            for d in range(horizon_days):
                cum_adds = [
                    a_var[(p.player_id, t)]
                    for p in free_agents
                    for t in range(d + 1)
                ]
                prob += (
                    pulp.lpSum([1 - r_var[(p.player_id, d)] for p in droppable_roster])
                    <= pulp.lpSum(cum_adds)
                )
        elif droppable_roster and not free_agents:
            for p in droppable_roster:
                for d in range(horizon_days):
                    prob += r_var[(p.player_id, d)] == 1

        for d in range(horizon_days):
            for slot in data.config.active_slots:
                slot_vars = [
                    x_var[(p.player_id, slot, d)]
                    for p in all_players
                    if (p.player_id, slot, d) in x_var
                ]
                if slot_vars:
                    prob += pulp.lpSum(slot_vars) <= 1

        for p in all_players:
            pid = p.player_id
            for d in range(horizon_days):
                if cm.has_game(pid, d):
                    p_day_vars = [
                        x_var[(pid, slot, d)]
                        for slot in cm.eligible_slots_map[pid]
                        if (pid, slot, d) in x_var
                    ]
                    if p_day_vars:
                        prob += pulp.lpSum(p_day_vars) <= r_var[(pid, d)]

        w1_add_vars = [
            a_var[(p.player_id, d)]
            for p in free_agents
            for d in data.horizon.week1_days
        ]
        if w1_add_vars:
            prob += pulp.lpSum(w1_add_vars) <= data.rem_acquisitions_w1

        w2_add_vars = [
            a_var[(p.player_id, d)]
            for p in free_agents
            for d in data.horizon.week2_days
        ]
        if w2_add_vars:
            prob += pulp.lpSum(w2_add_vars) <= data.cap_acquisitions_w2

        # Use HiGHS if available, else standard CBC
        solver = None
        for s_candidate in [
            lambda: pulp.PULP_CBC_CMD(msg=False),
            lambda: pulp.HiGHS(msg=False),
            lambda: pulp.CPSAT(msg=False),
        ]:
            try:
                solver = s_candidate()
                if solver.available():
                    break
            except Exception:
                continue

        if solver is None or isinstance(solver, pulp.CPSAT):
            return self._solve_ortools(data)

        t_solve_0 = time.perf_counter()
        status_code = prob.solve(solver)
        solver_wall_ms = (time.perf_counter() - t_solve_0) * 1000.0

        # Check for optimality across pulp versions
        is_optimal = False
        if hasattr(status_code, "status"):
            try:
                is_optimal = (int(status_code.status) == 1 or getattr(status_code, "has_solution", False))
            except Exception:
                is_optimal = True
        elif hasattr(status_code, "status_str"):
            is_optimal = (status_code.status_str.lower() == "optimal")
        elif isinstance(status_code, int):
            is_optimal = (status_code == 1)
        elif hasattr(prob, "status"):
            is_optimal = (prob.status == 1)
        else:
            is_optimal = True

        if not is_optimal:
            return StreamingOptimizationResult(
                status="INFEASIBLE",
                solve_time_ms=solver_wall_ms,
                total_time_ms=0.0,
                total_projected_fp=0.0,
                net_objective_value=0.0,
                total_drop_penalties=0.0,
                total_adds_week1=0,
                total_adds_week2=0,
            )

        res = self._extract_solution(
            data=data,
            cm=cm,
            status_str="OPTIMAL",
            get_r_val=lambda pid, d: pulp.value(r_var[(pid, d)]) > 0.5,
            get_a_val=lambda pid, d: pulp.value(a_var[(pid, d)]) > 0.5,
            get_x_val=lambda pid, s, d: (
                pulp.value(x_var[(pid, s, d)]) > 0.5
                if (pid, s, d) in x_var
                else False
            ),
        )
        res.solve_time_ms = solver_wall_ms
        return res

    def _extract_solution(
        self,
        data: OptimizationProblemData,
        cm: ConstraintManager,
        status_str: str,
        get_r_val,
        get_a_val,
        get_x_val,
    ) -> StreamingOptimizationResult:
        """Parse raw variable values into structured recommendations and metrics."""
        horizon_days = data.horizon.total_days
        last_day = horizon_days - 1
        all_players = data.all_players
        free_agents = data.candidate_free_agents
        droppable_roster = data.droppable_roster

        daily_plans: List[DailyLineupPlan] = []
        actions: List[StreamingAction] = []
        total_projected_fp = 0.0

        # Map day -> {slot: starter_str}, day -> assigned_pids, day -> day_fp, day -> active_games
        daily_starters: Dict[int, Dict[str, str]] = {d: {} for d in range(horizon_days)}
        daily_assigned: Dict[int, Set[int]] = {d: set() for d in range(horizon_days)}
        daily_active_games: Dict[int, int] = {d: 0 for d in range(horizon_days)}
        daily_fps: Dict[int, float] = {d: 0.0 for d in range(horizon_days)}

        for p in all_players:
            pid = p.player_id
            for d in range(horizon_days):
                if cm.has_game(pid, d):
                    for slot in cm.eligible_slots_map[pid]:
                        if get_x_val(pid, slot, d):
                            daily_starters[d][slot] = f"{p.name} ({p.team} - {p.fppg:.1f})"
                            daily_assigned[d].add(pid)
                            daily_active_games[d] += 1
                            daily_fps[d] += p.fppg

        for d in range(horizon_days):
            dt_str = str(data.horizon.get_date(d))
            week_num = data.horizon.get_week_of_day(d)
            day_fp = daily_fps[d]
            active_games = daily_active_games[d]
            starters = daily_starters[d]

            rostered_today_ids = {
                p.player_id for p in all_players if get_r_val(p.player_id, d)
            }
            bench_ids = rostered_today_ids - daily_assigned[d]
            bench_names = [
                f"{cm.player_map[pid].name} ({cm.player_map[pid].team})"
                for pid in bench_ids
            ]

            total_projected_fp += day_fp
            daily_plans.append(
                DailyLineupPlan(
                    day_idx=d,
                    date_str=dt_str,
                    week_num=week_num,
                    starters=starters,
                    bench=bench_names,
                    total_projected_fp=day_fp,
                    active_games_count=active_games,
                )
            )

        # Detect additions
        adds_w1 = 0
        adds_w2 = 0
        for p in free_agents:
            pid = p.player_id
            for d in range(horizon_days):
                if get_a_val(pid, d):
                    week = data.horizon.get_week_of_day(d)
                    if week == 1:
                        adds_w1 += 1
                    else:
                        adds_w2 += 1

                    actions.append(
                        StreamingAction(
                            day_idx=d,
                            date_str=str(data.horizon.get_date(d)),
                            action_type="ADD",
                            player_id=pid,
                            player_name=p.name,
                            player_team=p.team,
                            fppg=p.fppg,
                            reason=f"Add streamer in Week {week} for starting lineup value",
                        )
                    )

        # Detect drops from droppable initial roster
        retained_bench_holds = []
        dropped_bench_holds = []
        total_drop_penalties = 0.0

        for p in droppable_roster:
            pid = p.player_id
            was_retained = get_r_val(pid, last_day)
            if was_retained:
                retained_bench_holds.append(p.name)
            else:
                dropped_bench_holds.append(p.name)
                total_drop_penalties += p.drop_penalty_c_p
                # Find day dropped
                for d in range(horizon_days):
                    if not get_r_val(pid, d):
                        actions.append(
                            StreamingAction(
                                day_idx=d,
                                date_str=str(data.horizon.get_date(d)),
                                action_type="DROP",
                                player_id=pid,
                                player_name=p.name,
                                player_team=p.team,
                                fppg=p.fppg,
                                reason=f"Dropped bench hold (opportunity cost penalty: {p.drop_penalty_c_p:.1f})",
                            )
                        )
                        break

        # Sort actions chronologically
        actions.sort(key=lambda a: a.day_idx)

        net_objective = total_projected_fp - total_drop_penalties - (
            (adds_w1 + adds_w2) * data.add_friction_epsilon
        )

        return StreamingOptimizationResult(
            status=status_str,
            solve_time_ms=0.0,
            total_projected_fp=total_projected_fp,
            net_objective_value=net_objective,
            total_drop_penalties=total_drop_penalties,
            total_adds_week1=adds_w1,
            total_adds_week2=adds_w2,
            actions=actions,
            daily_plans=daily_plans,
            retained_bench_holds=retained_bench_holds,
            dropped_bench_holds=dropped_bench_holds,
        )
