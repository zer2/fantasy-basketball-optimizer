"""The same draft, played out in real weeks -- and the gap gets much wider.

Identical to team_differential.py in every respect but one: a drafted player no longer
contributes their season average, they contribute one week they actually played. Same pool,
same dealing, same histogram. The only thing that changes is that a player is allowed to have
a good week and a bad one.

That single change roughly doubles the spread. Season averages put the standard deviation of a
matchup at 30 points per game, about 91 across a week; real weeks put it at 137. The extra
comes from the variation an average erases -- a 25-point-per-game scorer who drops 12 one week
and 38 the next reads as "25" in every simulation of the other scene. It is the difference
between how good a team is and what it does in the week that counts, and it is the reason a
favourite still loses.

Weekly totals come from WEEKLY_NUMBERS_VIEW for 2025-26, and a player is only ever dealt a week
they genuinely have: weeks they missed are absent from the data rather than counted as zero.

Render one act while tuning it, the whole thing when it is right:

    manim -ql visualizations/scenes/weekly_differential.py WeeklyActOneSingleDraw
    manim -qh visualizations/scenes/weekly_differential.py WeeklyDifferentialFull
"""

from __future__ import annotations

import numpy as np

from differential_base import DifferentialSceneBase


class WeeklyDifferential(DifferentialSceneBase):
    """The real-weeks cut: a player is worth one week they played, drawn afresh each time."""

    data_filename      = 'weekly_2025_26.json'
    # Four and a half times the axis of the averages scene, because the spread really is that
    # much larger: weekly totals run about three times a per-game number, and real weeks add
    # half again on top of that. Three standard deviations either way, as there.
    differential_limit = 420
    bin_width          = 20
    axis_tick_step     = 140
    total_caption      = 'Points in the week'
    spread_caption     = 'standard deviation: {spread:.0f} points in the week'
    # Once the dealing is done the faces have nothing left to say, so they clear out and the
    # chart takes the full frame for the curve and the algebra.
    dismiss_rosters_after_montage = True

    def contribution_values(self, simulation_index: int) -> np.ndarray:
        """Each drafted player's dealt week, recovered so act one's totals climb in step.

        The prepared data stores team totals but not the individual weeks behind them, so the
        per-player split is reconstructed here by apportioning each side's total across its
        thirteen players in proportion to their season averages. Act one uses this only to
        animate the totals climbing as faces land; both sides still finish on exactly the
        stored total, so nothing downstream can disagree with the histogram.
        """
        roster = self.prepared['rosters'][simulation_index]
        season_averages = np.array(
            [player[self.prepared['value_key']] for player in self.prepared['pool']])
        drawn_averages = season_averages[roster]

        contributions = np.empty(len(roster))
        for side_index in range(2):
            side = slice(side_index * self.team_size, (side_index + 1) * self.team_size)
            share = drawn_averages[side] / drawn_averages[side].sum()
            contributions[side] = share * self.prepared['totals'][simulation_index][side_index]
        return contributions


class WeeklyActOneSingleDraw(WeeklyDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()


class WeeklyActThreeMontage(WeeklyDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
        self.play_act_two_repeated_draws()
        self.play_act_three_montage()


class WeeklyDifferentialFull(WeeklyDifferential):
    def construct(self) -> None:
        self.play_all_acts()
