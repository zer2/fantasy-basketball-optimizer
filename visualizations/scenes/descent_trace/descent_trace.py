"""Any one recorded gradient descent, as the app ran it: the seeds, the one chosen, then every iteration.

Machinery, not an explainer: no narration, and nothing here is specific to one candidate. It draws whatever
prepare_descent_trace_data.py recorded, named by the DESCENT_TRACE environment variable (a file in prepared_data/):

    python visualizations/scenes/descent_trace/prepare_descent_trace_data.py --rounds 3 --player "Ausar Thompson"
    DESCENT_TRACE=descent_trace_2025_26_round3_ausar_thompson.json manim -ql visualizations/scenes/descent_trace/descent_trace.py DescentTrace

Two acts. The seeds: every seed the app scored before descending, with its H-score, and the best of them chosen --
all of them, so the chosen start is on screen before the numbers start
moving. The descent: per iteration, the category weights (100 = neutral) with the expand view's G-score breakdown
beneath them, column for column; the H-score as a number and a small line; each category's win rate against each
opponent as a grid; and the roster's slot assignment with the players' faces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from manim import (
    DOWN, LEFT, RIGHT, UP, UL, UR, Axes, Circle, Create, DashedLine, FadeIn, FadeOut, GREY_B, GREY_D, GREY_E,
    Group, ImageMobject, Indicate, Line, Rectangle, RoundedRectangle, Scene, Text, Transform, VGroup, WHITE, YELLOW,
    BLUE_C, ManimColor, interpolate_color,
)

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent.parent
_PREPARED_DATA_DIR = _VISUALIZATIONS_DIR / 'prepared_data'
_HEADSHOT_DIR = _VISUALIZATIONS_DIR / 'prepared_assets' / 'headshots'

CATEGORY_LABELS = {'Field Goal %': 'FG%', 'Free Throw %': 'FT%', 'Threes': '3s', 'Points': 'PTS',
                   'Rebounds': 'REB', 'Assists': 'AST', 'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TO'}
SEEDS_PER_ROW = 6
STEP_RUN_TIME = 0.28                 # seconds per iteration

# Top right: the H-score's line, small (its shape is the point, not its values), left of the number.
H_AXES_CENTRE = (1.25, 3.25)
H_AXES_SIZE = (1.5, 0.5)

# Left: the category weights, and beneath them the G-score breakdown, one column per category.
WEIGHT_BAR_WIDTH = 0.42
WEIGHT_BAR_GAP = 0.16
WEIGHT_BASELINE = (-5.55, -0.1)      # left end of the weight bars' baseline
WEIGHT_BAR_HEIGHT_LIMIT = 2.0        # the tallest bar's height, at most: above it are the heading and the title
WEIGHT_BAR_SCALE = 0.0125            # scene units per display weight point (100 = neutral -> 1.25 units), unless
                                     # the descent's largest weight would pass the limit
DIFF_TOP = -0.48                     # top of the breakdown's first row
DIFF_ROW_HEIGHT = 0.3

# Right: each category's win rate against each opponent; categories are columns, as on the left.
GRID_TOP_LEFT = (1.1, 2.05)          # top-left corner of the first cell
GRID_CELL = (0.63, 0.3)              # width, height: nine columns end inside the frame's right edge
LOSING, EVEN, WINNING = ManimColor('#b8433f'), ManimColor('#3a3a46'), ManimColor('#3f9a5a')

# The breakdown's colour reaches full strength at the app's own scales (styler_functions.ts): a player's G-scores at
# about +-1.8, the team-level rows (sums and differentials of several players) at about +-6.
PLAYER_FULL_COLOUR = 1.8
TEAM_FULL_COLOUR = 6.0

# Bottom: the roster's slots, with the players' faces.
ROSTER_Y = -3.25                     # the faces' centre
ROSTER_SLOT_WIDTH = 1.04
PORTRAIT_DIAMETER = 0.5


def load_trace() -> dict:
    """The recording named by DESCENT_TRACE: there is no default, since a scene of the wrong descent is worse than
    none."""
    name = os.environ.get('DESCENT_TRACE')
    if not name:
        raise SystemExit('Set DESCENT_TRACE to a recording in visualizations/prepared_data/ '
                         '(see prepare_descent_trace_data.py).')
    return json.loads((_PREPARED_DATA_DIR / name).read_text(encoding='utf-8'))


def label_seed(seed: dict) -> str:
    if seed['kind'] == 'gentle punt':
        return f"{CATEGORY_LABELS[seed['category']]} at 50%"
    if seed['kind'] == 'balanced':
        return 'balanced'
    raise ValueError(f"Unknown seed kind {seed['kind']!r}: re-record the trace with the current recorder.")


def colour_by_sign(value: float, middle: float, full_colour: float):
    """Red below the middle, grey at it, green above, at full strength `full_colour` away from it."""
    if value < middle:
        return interpolate_color(EVEN, LOSING, min((middle - value) / full_colour, 1.0))
    return interpolate_color(EVEN, WINNING, min((value - middle) / full_colour, 1.0))


def shorten_team(team: str) -> str:
    """'Drafter 7' -> 'D7': the grid's row headings have no room for more."""
    words = team.split()
    return words[0][0] + words[-1] if len(words) > 1 and words[-1].isdigit() else team[:4]


def fit_text_width(text: Text, width: float) -> Text:
    """Shrinks `text` to `width` when it is wider."""
    if text.width > width:
        text.scale(width / text.width)
    return text


class DescentTraceDrawing:
    """Draws one recorded descent: mixed into a Scene (or a ThreeDScene facing straight down, which draws flat mobjects
    as a Scene does). Call use_trace first; DescentTrace plays all of it, and another scene can play parts -- the start
    card instead of the seeds, the steps paced to its narration."""

    def use_trace(self, trace: dict) -> None:
        self.trace = trace
        self.categories = [CATEGORY_LABELS[category] for category in trace['categories']]
        largest_weight = max(max(entry['weights']) for entry in trace['iterations'])
        self.weight_bar_scale = min(WEIGHT_BAR_SCALE, WEIGHT_BAR_HEIGHT_LIMIT / largest_weight)

    def find_column_centre(self, position: int) -> float:
        """The x of a category's column on the left: its weight bar and its breakdown cells."""
        return WEIGHT_BASELINE[0] + position * (WEIGHT_BAR_WIDTH + WEIGHT_BAR_GAP) + WEIGHT_BAR_WIDTH / 2

    def find_total_column_centre(self) -> float:
        """The x of the breakdown's Total column, one column past the last category."""
        return self.find_column_centre(len(self.categories)) + 0.08

    # ── The title ──────────────────────────────────────────────────────────────

    def play_title(self) -> None:
        trace = self.trace
        moment = 'first pick' if trace['rounds'] == 0 else f"after {trace['rounds']} rounds"
        format_part = ', Rotisserie' if trace['scoring_format'] == 'Rotisserie' else ''
        self.title = Text(f"{trace['player']}  ·  {trace['season']}{format_part}, {moment}",
                          font_size=22, color=WHITE).to_corner(UL, buff=0.35)
        self.play(FadeIn(self.title), run_time=0.8)

    # ── Act 1: the seeds ───────────────────────────────────────────────────────

    def play_start_card(self) -> None:
        """The start's own card, top right, where the chosen seed's card goes after the seeds: its name and score."""
        start = self.trace['start']
        box = RoundedRectangle(width=1.95, height=1.05, corner_radius=0.12, stroke_color=YELLOW, stroke_width=4)
        content = VGroup(Text(start['label'], font_size=17, color=WHITE),
                         Text(f"{start['h_score']:.2f}%", font_size=24, color=WHITE)).arrange(DOWN, buff=0.12)
        self.chosen_card = VGroup(box, content.move_to(box)).scale(0.75).to_corner(UR, buff=0.35)
        self.play(FadeIn(self.chosen_card), run_time=0.6)

    def play_seeds(self) -> None:
        seeds = self.trace['seeds']
        if not seeds:
            # No seed scan (Rotisserie starts every candidate from its one lowvar seed): the start's own card, where
            # the chosen seed's card would have gone.
            self.play_start_card()
            return
        chosen = self.trace['chosen_seed']
        heading = Text('The seeds, scored before any descent', font_size=24, color=GREY_B).move_to(UP * 1.6)
        cards = VGroup()
        for seed in seeds:
            box = RoundedRectangle(width=1.95, height=1.05, corner_radius=0.12, stroke_color=GREY_D,
                                   stroke_width=2)
            label = Text(label_seed(seed), font_size=17, color=WHITE)
            score = Text(f"{seed['h_score']:.2f}%", font_size=24, color=WHITE)
            content = VGroup(label, score).arrange(DOWN, buff=0.12).move_to(box)
            cards.add(VGroup(box, content))
        rows = [VGroup(*cards[start:start + SEEDS_PER_ROW]).arrange(RIGHT, buff=0.2)
                for start in range(0, len(cards), SEEDS_PER_ROW)]
        VGroup(*rows).arrange(DOWN, buff=0.3).next_to(heading, DOWN, buff=0.45)
        self.play(FadeIn(heading), FadeIn(cards, lag_ratio=0.06), run_time=1.6)
        self.wait(0.8)

        best = cards[chosen]
        best[0].set_stroke(YELLOW, width=4)
        others = VGroup(*[card for index, card in enumerate(cards) if index != chosen])
        verdict = Text('highest H-score: the descent starts here', font_size=20, color=YELLOW).next_to(
            VGroup(*rows), DOWN, buff=0.4)
        self.play(others.animate.set_opacity(0.3), Indicate(best, color=YELLOW, scale_factor=1.08),
                  FadeIn(verdict), run_time=1.2)
        self.wait(1.2)
        self.chosen_card = best
        self.play(FadeOut(others), FadeOut(heading), FadeOut(verdict),
                  best.animate.scale(0.75).to_corner(UR, buff=0.35), run_time=0.9)

    # ── Act 2: the descent ─────────────────────────────────────────────────────

    def play_descent(self) -> None:
        self.play_starting_state()
        self.wait(0.5)
        self.play_steps(range(1, len(self.trace['iterations'])), STEP_RUN_TIME)
        self.wait(2.0)

    def play_starting_state(self) -> None:
        """The frame, and the state before the first step."""
        self.build_static_frame()
        self.state = self.draw_state(0)
        self.roster = self.draw_roster(0)
        self.play(*[FadeIn(part) for part in self.state], FadeIn(self.roster), run_time=0.6)

    def play_steps(self, indices, step_run_time: float) -> None:
        """Each iteration in `indices` (consecutive, after the ones already shown), step_run_time seconds apiece."""
        iterations = self.trace['iterations']
        bars, cells, numbers = self.state
        for index in indices:
            new_bars, new_cells, new_numbers = self.draw_state(index)
            # The numbers change at once; only the bars, the cells' colours and the H-score line move. Transforming
            # text morphs one number's glyphs into the next, which is unreadable mid-step.
            self.remove(numbers)
            numbers = new_numbers
            self.add(numbers)
            # The faces are redrawn only when a player changes slot, which is rare; never faded or dimmed, since an
            # ImageMobject's alpha does not survive an opacity change (its circle squares off into a grey block).
            if iterations[index]['slots'] != iterations[index - 1]['slots']:
                self.remove(self.roster)
                self.roster = self.draw_roster(index)
                self.add(self.roster)
            self.play(Transform(bars, new_bars), Transform(cells, new_cells), self.grow_h_line(index),
                      run_time=step_run_time)
        self.state = (bars, cells, numbers)

    def build_static_frame(self) -> None:
        """What does not change: the headings, the weight bars' baseline, neutral line and category names (which head
        the breakdown's columns too), the breakdown's row names, the grid's headings, and the H-score axes."""
        left, base_y = WEIGHT_BASELINE
        span = len(self.categories) * (WEIGHT_BAR_WIDTH + WEIGHT_BAR_GAP) - WEIGHT_BAR_GAP
        baseline = Line((left, base_y, 0), (left + span, base_y, 0), color=GREY_D)
        neutral_y = base_y + 100 * self.weight_bar_scale
        neutral = DashedLine((left, neutral_y, 0), (left + span, neutral_y, 0), color=GREY_B, stroke_width=2)
        neutral_label = Text('neutral', font_size=13, color=GREY_B).next_to(neutral, RIGHT, buff=0.08)
        names = VGroup(*[Text(label, font_size=15, color=GREY_B).move_to(
            (self.find_column_centre(position), base_y - 0.2, 0)) for position, label in enumerate(self.categories)])
        total_name = Text('Total', font_size=15, color=GREY_B).move_to((self.find_total_column_centre(), base_y - 0.2, 0))
        weights_heading = Text('Category weights', font_size=19, color=GREY_B).move_to(
            (left + span / 2, GRID_TOP_LEFT[1] + 0.42, 0))
        # The breakdown's rows never reorder, so their names are drawn once, from the first iteration.
        diff_names = VGroup()
        for row, entry in enumerate(self.trace['iterations'][0]['diff_rows']):
            name = fit_text_width(Text(entry['label'], font_size=13, color=YELLOW if entry['is_candidate'] else GREY_B),
                                  1.35)
            diff_names.add(name.move_to((left - 0.12, DIFF_TOP - (row + 0.5) * DIFF_ROW_HEIGHT, 0),
                                        aligned_edge=RIGHT))

        grid_left, grid_top = GRID_TOP_LEFT
        cell_width, cell_height = GRID_CELL
        opponents = self.trace['opponents']
        grid_heading = Text('Win rate in each category against each opponent', font_size=19, color=GREY_B).move_to(
            (grid_left + len(self.categories) * cell_width / 2, grid_top + 0.42, 0))
        column_heads = VGroup(*[Text(label, font_size=13, color=GREY_B).move_to(
            (grid_left + (column + 0.5) * cell_width, grid_top + 0.15, 0)) for column, label in enumerate(self.categories)])
        row_heads = VGroup(*[Text(shorten_team(team), font_size=12, color=GREY_B).move_to(
            (grid_left - 0.08, grid_top - (row + 0.5) * cell_height - 0.05, 0), aligned_edge=RIGHT)
            for row, team in enumerate(opponents)])

        n_iterations = len(self.trace['iterations'])
        scores = [entry['h_score'] for entry in self.trace['iterations']]
        low, high = min(scores), max(scores)
        pad = max((high - low) * 0.15, 0.01)
        self.h_axes = Axes(x_range=(1, n_iterations, n_iterations - 1), y_range=(low - pad, high + pad, high - low + 2 * pad),
                           x_length=H_AXES_SIZE[0], y_length=H_AXES_SIZE[1], tips=False,
                           axis_config={'color': GREY_D, 'stroke_width': 1.5, 'include_ticks': False}
                           ).move_to((*H_AXES_CENTRE, 0))
        # A Rotisserie H-score approximates the chance of winning the whole league (around 8% on average), not a
        # matchup, so it says so.
        h_label = ('H-score (league win chance) by iteration' if self.trace['scoring_format'] == 'Rotisserie'
                   else 'H-score by iteration')
        h_heading = Text(h_label, font_size=11, color=GREY_B).next_to(self.h_axes, DOWN, buff=0.06)
        roster_heading = Text('Roster slots', font_size=15, color=GREY_B).move_to(
            (-6.55, ROSTER_Y + 0.72, 0), aligned_edge=LEFT)
        self.h_line = VGroup()
        self.play(FadeIn(VGroup(baseline, neutral, neutral_label, names, total_name, weights_heading, diff_names,
                                grid_heading, column_heads, row_heads, h_heading, roster_heading)),
                  Create(self.h_axes), run_time=0.8)
        self.add(self.h_line)

    def draw_state(self, index: int) -> tuple[VGroup, VGroup, VGroup]:
        """What changes, after iteration `index`: the weight bars, the breakdown's and the grid's cells, and every
        number."""
        entry = self.trace['iterations'][index]
        _, base_y = WEIGHT_BASELINE
        bars, cells, numbers = VGroup(), VGroup(), VGroup()

        counter = Text(f"iteration {entry['iteration']} of {len(self.trace['iterations'])}", font_size=18,
                       color=GREY_B).next_to(self.chosen_card, LEFT, buff=0.35).shift(UP * 0.2)
        h_score = Text(f"H-score {entry['h_score']:.2f}%", font_size=28, color=YELLOW).next_to(
            counter, DOWN, buff=0.1, aligned_edge=RIGHT)
        numbers.add(counter, h_score)

        for position, weight in enumerate(entry['weights']):
            height = max(weight * self.weight_bar_scale, 0.001)
            bar = Rectangle(width=WEIGHT_BAR_WIDTH, height=height, stroke_width=0, fill_color=BLUE_C,
                            fill_opacity=0.85).move_to((self.find_column_centre(position), base_y + height / 2, 0))
            bars.add(bar)
            numbers.add(Text(f'{weight:.0f}', font_size=14, color=WHITE).next_to(bar, UP, buff=0.06))

        diff_cell_width = WEIGHT_BAR_WIDTH + WEIGHT_BAR_GAP - 0.04
        for row, diff_row in enumerate(entry['diff_rows']):
            full_colour = PLAYER_FULL_COLOUR if diff_row['is_candidate'] else TEAM_FULL_COLOUR
            y = DIFF_TOP - (row + 0.5) * DIFF_ROW_HEIGHT
            columns = [(self.find_column_centre(position), value) for position, value in enumerate(diff_row['values'])]
            # The Total column colours at the team scale whatever the row: it sums nine categories.
            columns.append((self.find_total_column_centre(), diff_row['total']))
            for column, (x, value) in enumerate(columns):
                scale = TEAM_FULL_COLOUR if column == len(self.categories) else full_colour
                cells.add(Rectangle(width=diff_cell_width, height=DIFF_ROW_HEIGHT - 0.04, stroke_width=0,
                                    fill_color=colour_by_sign(value, 0.0, scale), fill_opacity=1.0).move_to((x, y, 0)))
                numbers.add(Text(f'{value:.2f}', font_size=11, color=WHITE).move_to((x, y, 0)))

        grid_left, grid_top = GRID_TOP_LEFT
        cell_width, cell_height = GRID_CELL
        # Recorded categories x opponents; drawn opponents as rows, categories as columns.
        for column, category_rates in enumerate(entry['category_opponent_win_rates']):
            for row, rate in enumerate(category_rates):
                centre = (grid_left + (column + 0.5) * cell_width, grid_top - (row + 0.5) * cell_height - 0.05, 0)
                cells.add(Rectangle(width=cell_width - 0.04, height=cell_height - 0.04, stroke_width=0,
                                    fill_color=colour_by_sign(rate, 50.0, 40.0), fill_opacity=1.0).move_to(centre))
                numbers.add(Text(f'{rate:.0f}', font_size=11, color=WHITE).move_to(centre))
        return bars, cells, numbers

    def draw_roster(self, index: int) -> Group:
        """Every roster slot after iteration `index`: its name, and the face and last name of the player in it, the
        candidate ringed; an open slot (one the picks to come will fill) is an empty circle."""
        slots = self.trace['iterations'][index]['slots']
        players = self.trace['players']
        start_x = -(len(slots) - 1) * ROSTER_SLOT_WIDTH / 2
        roster = Group()
        for position, slot in enumerate(slots):
            x = start_x + position * ROSTER_SLOT_WIDTH
            is_candidate = slot['player_id'] == self.trace['player_id']
            roster.add(Text(slot['slot'], font_size=13, color=YELLOW if is_candidate else GREY_B).move_to(
                (x, ROSTER_Y + 0.42, 0)))
            ring = Circle(radius=PORTRAIT_DIAMETER / 2, stroke_color=YELLOW if is_candidate else GREY_D,
                          stroke_width=3 if is_candidate else 1.5, fill_color=GREY_E,
                          fill_opacity=1.0 if slot['player_id'] is not None else 0.0).move_to((x, ROSTER_Y, 0))
            roster.add(ring)
            if slot['player_id'] is None:
                continue
            portrait = ImageMobject(str(_HEADSHOT_DIR / f"{slot['player_id']}.png"))
            portrait.height = PORTRAIT_DIAMETER
            portrait.move_to(ring.get_center())
            name = fit_text_width(Text(players[str(slot['player_id'])]['last_name'], font_size=12,
                                       color=YELLOW if is_candidate else WHITE), ROSTER_SLOT_WIDTH - 0.08)
            roster.add(portrait, name.move_to((x, ROSTER_Y - 0.42, 0)))
        return roster

    def grow_h_line(self, index: int):
        """Extends the H-score line to iteration `index` (1-based iterations, so index + 1 on the axis)."""
        scores = [entry['h_score'] for entry in self.trace['iterations']]
        segment = Line(self.h_axes.c2p(index, scores[index - 1]), self.h_axes.c2p(index + 1, scores[index]),
                       color=YELLOW, stroke_width=2.5)
        self.h_line.add(segment)
        return Create(segment)


class DescentTrace(Scene, DescentTraceDrawing):

    def construct(self) -> None:
        self.use_trace(load_trace())
        self.play_title()
        self.play_seeds()
        self.play_descent()
