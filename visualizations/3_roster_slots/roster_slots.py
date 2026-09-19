"""Why the optimiser parks your centre in the cheapest slot he is eligible for.

Thirteen roster slots, thirteen rows: the five players already drafted, and the eight picks not
yet made. A drafted player scores 0 in every slot he can play and -inf in every slot he cannot,
so he is worth exactly nothing wherever he ends up. All the value in the matrix sits in the
future-pick rows, and because every slot gets used exactly once, maximising the total is the
same thing as making the drafted five occupy the CHEAPEST slots available to them. The optimiser
is not placing them to be useful. It is moving them out of the way.

The numbers come from `prepare_assignment_data.py`, which runs the app's own
`get_player_rows` / `get_future_player_rows` / `optimize_positions_all_players`. Run that once
first; this scene reads its JSON and does no solving of its own.

    manim -ql visualizations/scenes/assignment.py RosterSlotAssignment
    manim -qh visualizations/scenes/assignment.py RosterSlotAssignment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, Group, VGroup, ImageMobject, Rectangle, Circle, Line, Text, DecimalNumber,
    FadeIn, FadeOut, Create, Transform, Indicate,
    DOWN, RIGHT,
    YELLOW, WHITE, GREY_A, GREY_B, GREY_D, GREY_E, GREEN_B, RED_B, RED_D, BLUE_B, BLUE_E, BLACK,
)


from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from narration import NARRATION


# ── Where things sit (Manim's frame is 14.2 x 8 units, origin at the centre) ───────────
# Thirteen columns is the binding constraint: at any wider cell the grid runs off the right of
# the frame, and at any narrower one a signed three-decimal reward stops being legible.

CELL_WIDTH  = 0.78
CELL_HEIGHT = 0.44
GRID_CENTRE_X   = 1.42     # pushed right of centre to leave the left third for player names
GRID_TOP_ROW_Y  = 2.42

ROW_LABEL_RIGHT_EDGE = -3.96   # names and chips end here, clear of the grid's left edge
PORTRAIT_CENTRE_X    = -6.60
PORTRAIT_DIAMETER    = 0.38

SLOT_HEADER_Y  = 2.88
READOUT_CENTRE = (-5.30, 3.22, 0.0)

CELL_NUMBER_FONT   = 12
SLOT_HEADER_FONT   = 15
ROW_LABEL_FONT     = 15
# How long the frame holds where a line of narration goes. The scene carries no text of its
# own, so these pauses are the only room the voice has.

_DATA_PATH    = Path(__file__).resolve().parent.parent / 'data' / 'assignment_2025_26.json'
_HEADSHOT_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'headshots'


def load_assignment_data(data_path: Path) -> dict:
    """The matrix and the two assignments written by prepare_assignment_data.py."""
    if not data_path.exists():
        raise FileNotFoundError(
            f'{data_path} is missing. Run `python visualizations/prepare_assignment_data.py` '
            f'first -- the scene deliberately does no solving of its own.')
    prepared = json.loads(data_path.read_text(encoding='utf-8'))
    # JSON carries an ineligible cell as null; the scene wants the -inf it stands for, so that
    # "worth nothing here" and "cannot go here" stay different things all the way to the screen.
    prepared['reward_matrix'] = np.array(
        [[-np.inf if value is None else value for value in row]
         for row in prepared['reward_matrix']])
    return prepared


def sort_future_rows_by_slot(assignment: list[int], drafted_count: int) -> list[int]:
    """The same assignment, with the future-pick rows sorted by the slot they took.

    Every future-pick row holds an identical reward vector, so which future row sits in which
    leftover slot is arbitrary and the solver's answer for it is whatever the Hungarian algorithm
    happened to walk into. Sorting them gives the scene a stable starting picture instead of one
    whose row order carries a meaning it does not have. The drafted rows -- the only ones where
    the identity of the row matters -- are left exactly as solved.
    """
    return list(assignment[:drafted_count]) + sorted(assignment[drafted_count:])


def align_future_rows_to(
    reference: list[int]
    , assignment: list[int]
    , drafted_count: int
) -> list[int]:
    """`assignment`, with its future-pick rows arranged to sit still where `reference` already had them.

    Again exploiting that the future rows are interchangeable: a future row whose slot survives
    into this assignment keeps it, and only the slots that genuinely changed hands get handed to
    new rows. Without this the two assignments differ in almost every future row -- every marker
    slides one column along -- and the single move the scene is arguing about is lost in the
    shuffle. With it, moving a drafted player to a cheaper slot shows up as exactly what it is:
    that player steps down, and one future pick steps into the slot he vacated.
    """
    future_slots = set(assignment[drafted_count:])
    aligned = list(assignment[:drafted_count]) + [
        slot if slot in future_slots else None for slot in reference[drafted_count:]
    ]
    unclaimed = sorted(future_slots - set(aligned[drafted_count:]))
    for row_index in range(drafted_count, len(aligned)):
        if aligned[row_index] is None:
            aligned[row_index] = unclaimed.pop(0)
    return aligned


class RosterSlotAssignment(VoiceoverScene):
    """Build the matrix, state the constraint, guess wrong, then solve it."""

    def setup(self) -> None:
        self.prepared = load_assignment_data(_DATA_PATH)
        self.slot_labels = self.prepared['slot_labels']
        self.slot_count = len(self.slot_labels)
        self.drafted_players = self.prepared['drafted_players']
        self.drafted_count = len(self.drafted_players)
        self.future_pick_count = self.prepared['future_pick_count']
        self.future_pick_row = self.prepared['future_pick_row']
        self.reward_matrix = self.prepared['reward_matrix']

        # The naive assignment is shown first, so it sets the row order and the optimal one is
        # arranged around it rather than the other way round.
        self.naive_assignment = sort_future_rows_by_slot(
            self.prepared['naive_assignment'], self.drafted_count)
        self.optimal_assignment = align_future_rows_to(
            self.naive_assignment, self.prepared['optimal_assignment'], self.drafted_count)

        # The row the two assignments disagree about is the whole argument, so it is found rather
        # than named: a different five players would move a different row and the scene should
        # still point at the right one.
        self.decisive_row = next(
            row for row in range(self.drafted_count)
            if self.optimal_assignment[row] != self.naive_assignment[row])

        self.largest_reward_magnitude = max(abs(value) for value in self.future_pick_row)
        self.markers: dict[int, Rectangle] = {}

    # ── Geometry ──────────────────────────────────────────────────────────────────────

    def _cell_centre(self, row_index: int, column_index: int) -> np.ndarray:
        return np.array([
            GRID_CENTRE_X + (column_index - (self.slot_count - 1) / 2) * CELL_WIDTH,
            GRID_TOP_ROW_Y - row_index * CELL_HEIGHT,
            0.0,
        ])

    def _column_centre_x(self, column_index: int) -> float:
        return float(self._cell_centre(0, column_index)[0])

    # ── Cells ─────────────────────────────────────────────────────────────────────────

    def _build_cell_rectangle(self
                              , row_index: int
                              , column_index: int
                              , fill_colour
                              , fill_opacity: float) -> Rectangle:
        rectangle = Rectangle(
            width        = CELL_WIDTH * 0.94,
            height       = CELL_HEIGHT * 0.88,
            fill_color   = fill_colour,
            fill_opacity = fill_opacity,
            stroke_color = GREY_D,
            stroke_width = 0.8,
        )
        rectangle.move_to(self._cell_centre(row_index, column_index))
        return rectangle

    def _build_drafted_cell(self, row_index: int, column_index: int) -> VGroup:
        """One cell of a drafted player's row: a zero where he is eligible, -inf where he is not.

        The zero is the point of the whole scene and so it is written out rather than implied by
        a colour: a player you have already drafted adds nothing to the objective in any slot he
        can legally occupy, which is why the optimiser has no reason to prefer one of them.
        """
        is_eligible = self.drafted_players[row_index]['eligible_slots'][column_index]
        cell = VGroup(self._build_cell_rectangle(
            row_index, column_index,
            BLUE_E if is_eligible else BLACK,
            0.45 if is_eligible else 1.0,
        ))
        cell.add(Text(
            '0' if is_eligible else '−∞',
            font_size = CELL_NUMBER_FONT + (2 if is_eligible else 0),
            color     = WHITE if is_eligible else GREY_D,
        ).move_to(self._cell_centre(row_index, column_index)))
        return cell

    def _build_future_cell(self, row_index: int, column_index: int) -> VGroup:
        """One cell of a future-pick row: what a pick you have not made is worth in this slot.

        Shaded by sign and magnitude because the ORDER of the thirteen is what has to be legible
        at a glance -- the scene's argument is that a drafted player should be pushed into a red
        slot so that a green one survives for someone who can actually use it.
        """
        value = self.future_pick_row[column_index]
        cell = VGroup(self._build_cell_rectangle(
            row_index, column_index,
            GREEN_B if value > 0 else RED_B,
            0.12 + 0.50 * abs(value) / self.largest_reward_magnitude,
        ))
        cell.add(Text(
            f'{value:+.3f}', font_size=CELL_NUMBER_FONT, color=WHITE,
        ).move_to(self._cell_centre(row_index, column_index)))
        return cell

    # ── Row labels ────────────────────────────────────────────────────────────────────

    def _build_drafted_row_label(self, row_index: int) -> Group:
        """A circular headshot and the player's name with the positions he is eligible at."""
        player = self.drafted_players[row_index]
        centre_y = float(self._cell_centre(row_index, 0)[1])

        backing = Circle(
            radius       = PORTRAIT_DIAMETER / 2,
            fill_color   = GREY_E,
            fill_opacity = 1.0,
            stroke_color = BLUE_B,
            stroke_width = 1.6,
        ).move_to([PORTRAIT_CENTRE_X, centre_y, 0.0])
        portrait = ImageMobject(str(_HEADSHOT_DIR / f'{player["player_id"]}.png'))
        portrait.height = PORTRAIT_DIAMETER
        portrait.move_to(backing.get_center())

        written = VGroup(
            Text(player['name'], font_size=ROW_LABEL_FONT, color=WHITE),
            Text('/'.join(player['eligible_positions']), font_size=ROW_LABEL_FONT - 3,
                 color=GREY_B),
        ).arrange(RIGHT, buff=0.16)
        # Scaled only when it would otherwise run into the grid: a four-position name like
        # "Scottie Barnes C/PF/SG/SF" is half again as wide as "Nikola Jokic C", and shrinking
        # every label to fit the longest would waste the width the short ones do not need.
        available_width = ROW_LABEL_RIGHT_EDGE - (PORTRAIT_CENTRE_X + PORTRAIT_DIAMETER / 2 + 0.18)
        if written.width > available_width:
            written.scale(available_width / written.width)
        written.next_to(backing, RIGHT, buff=0.18)

        return Group(backing, portrait, written)

    def _build_future_row_label(self, future_index: int) -> VGroup:
        """A pick that has not happened yet: no face, just its number in the draft order."""
        centre_y = float(self._cell_centre(self.drafted_count + future_index, 0)[1])
        marker = Circle(
            radius       = PORTRAIT_DIAMETER / 2,
            fill_color   = GREY_E,
            fill_opacity = 1.0,
            stroke_color = GREY_D,
            stroke_width = 1.6,
        ).move_to([PORTRAIT_CENTRE_X, centre_y, 0.0])
        label = VGroup(
            marker,
            Text('?', font_size=ROW_LABEL_FONT, color=GREY_B).move_to(marker.get_center()),
            Text(f'pick {self.drafted_count + future_index + 1}',
                 font_size=ROW_LABEL_FONT - 1, color=GREY_A).next_to(marker, RIGHT, buff=0.18),
        )
        return label

    # ── Static furniture ──────────────────────────────────────────────────────────────

    def build_grid_frame(self) -> None:
        """Slot headers across the top, row labels down the left, and the line between the halves."""
        self.slot_headers = VGroup(*[
            Text(label, font_size=SLOT_HEADER_FONT, color=GREY_A, weight='BOLD')
            .move_to([self._column_centre_x(column_index), SLOT_HEADER_Y, 0.0])
            for column_index, label in enumerate(self.slot_labels)
        ])

        self.drafted_row_labels = Group(*[
            self._build_drafted_row_label(row_index) for row_index in range(self.drafted_count)
        ])
        self.future_row_labels = VGroup(*[
            self._build_future_row_label(future_index)
            for future_index in range(self.future_pick_count)
        ])

        # The divider is not decoration: above it every row is worth zero, below it every row
        # carries the same reward vector, and that split is the fact the scene is built on.
        divider_y = float(self._cell_centre(self.drafted_count, 0)[1]) + CELL_HEIGHT / 2
        self.half_divider = Line(
            [PORTRAIT_CENTRE_X - PORTRAIT_DIAMETER, divider_y, 0.0],
            [self._column_centre_x(self.slot_count - 1) + CELL_WIDTH / 2, divider_y, 0.0],
            color=GREY_D, stroke_width=1.6,
        )

    # ── Act one: the matrix ───────────────────────────────────────────────────────────

    def play_act_one_eligibility(self) -> None:
        with self.voiceover(text=NARRATION['slots']):
            self.play(FadeIn(self.slot_headers), run_time=1.0)

        with self.voiceover(text=NARRATION['the_roster']):
            self.drafted_cells = {}
            for row_index in range(self.drafted_count):
                row_cells = VGroup(*[
                    self._build_drafted_cell(row_index, column_index)
                    for column_index in range(self.slot_count)
                ])
                for column_index in range(self.slot_count):
                    self.drafted_cells[(row_index, column_index)] = row_cells[column_index]
                self.play(
                    FadeIn(self.drafted_row_labels[row_index], shift=RIGHT * 0.2),
                    FadeIn(row_cells),
                    run_time=0.55,
                )
            self.wait(0.6)

    # ── Act two: where the value actually is ──────────────────────────────────────────

    def play_act_two_future_rewards(self) -> None:
        with self.voiceover(text=NARRATION['future_picks']):
            self.play(Create(self.half_divider), run_time=0.6)

        with self.voiceover(text=NARRATION['future_rows']):
            self.future_cells = {}
            for future_index in range(self.future_pick_count):
                row_index = self.drafted_count + future_index
                row_cells = VGroup(*[
                    self._build_future_cell(row_index, column_index)
                    for column_index in range(self.slot_count)
                ])
                for column_index in range(self.slot_count):
                    self.future_cells[(row_index, column_index)] = row_cells[column_index]
                self.play(
                    FadeIn(self.future_row_labels[future_index], shift=RIGHT * 0.2),
                    FadeIn(row_cells),
                    run_time=0.42 if future_index < 2 else 0.22,
                )
            self.wait(0.8)

        best_column = int(np.argmax(self.future_pick_row))
        with self.voiceover(text=NARRATION['best_slot']):
            self.play(
                *[Indicate(self.future_cells[(self.drafted_count + future_index, best_column)],
                           color=GREEN_B, scale_factor=1.12)
                  for future_index in range(self.future_pick_count)],
                run_time=1.0,
            )

    # ── Act three: what an assignment is allowed to be ────────────────────────────────

    def _build_marker(self, row_index: int, column_index: int) -> Rectangle:
        """The ring that says "this row is using this slot"."""
        marker = Rectangle(
            width        = CELL_WIDTH * 0.94,
            height       = CELL_HEIGHT * 0.88,
            stroke_color = YELLOW,
            stroke_width = 3.0,
            fill_opacity = 0.0,
        )
        marker.move_to(self._cell_centre(row_index, column_index))
        return marker

    def play_act_three_constraint(self) -> None:
        with self.voiceover(text=NARRATION['one_slot_each']):
            for row_index, column_index in enumerate(self.naive_assignment):
                self.markers[row_index] = self._build_marker(row_index, column_index)
                self.play(Create(self.markers[row_index]),
                          run_time=0.30 if row_index < 5 else 0.13)
            self.wait(0.8)

        # The violation to show is found rather than staged: the first drafted player who is
        # eligible for a slot another drafted player has already taken. With two centre-only
        # players on the roster that is one centre trying to stand in the other's slot, which is
        # the clash a viewer will already have half-noticed.
        offending_row, occupied_row = next(
            (row, other)
            for row in range(self.drafted_count)
            for other in range(self.drafted_count)
            if other != row
            and self.drafted_players[row]['eligible_slots'][self.naive_assignment[other]]
        )
        contested_column = self.naive_assignment[occupied_row]
        legal_column = self.naive_assignment[offending_row]

        with self.voiceover(text=NARRATION['the_clash']):
            self.play(self.markers[offending_row].animate.move_to(
                self._cell_centre(offending_row, contested_column)), run_time=0.7)
            self.play(
                self.markers[offending_row].animate.set_stroke(RED_D),
                self.markers[occupied_row].animate.set_stroke(RED_D),
                run_time=0.4,
            )

            rejection = VGroup(*[
                Line(self._cell_centre(row, contested_column)
                     + np.array([-0.3, -0.16 * sign, 0.0]),
                     self._cell_centre(row, contested_column)
                     + np.array([0.3, 0.16 * sign, 0.0]),
                     color=RED_D, stroke_width=4)
                for row in (offending_row, occupied_row) for sign in (-1, 1)
            ])
            self.play(Create(rejection), run_time=0.5)
            self.wait(0.6)
            # The repair belongs to this line too: the sentence ends on "not permissible", and
            # leaving the illegal board on screen after that would contradict it.
            self.play(FadeOut(rejection), run_time=0.3)
            self.play(
                self.markers[offending_row].animate.move_to(
                    self._cell_centre(offending_row, legal_column)).set_stroke(YELLOW),
                self.markers[occupied_row].animate.set_stroke(YELLOW),
                run_time=0.6,
            )

    # ── Act four: the assignment a person would guess ─────────────────────────────────

    def _build_total_readout(self) -> VGroup:
        """The one number the whole grid is competing to raise, in a box of its own.

        Boxed rather than captioned because the scene is narrated: a frame of numbers needs one
        place the eye knows to return to, and a border does that without a sentence explaining
        what it is.
        """
        self.total_number = DecimalNumber(
            self.prepared['naive_total'], num_decimal_places=3, font_size=44, color=YELLOW)
        label = Text('total', font_size=17, color=GREY_B)
        contents = VGroup(label, self.total_number).arrange(DOWN, buff=0.10)

        box = Rectangle(
            width=contents.width + 0.55, height=contents.height + 0.45,
            stroke_color=GREY_B, stroke_width=2, fill_color=BLACK, fill_opacity=0.9,
        ).move_to(contents)

        readout = VGroup(box, contents)
        readout.move_to(READOUT_CENTRE)
        return readout

    def play_act_four_naive_assignment(self) -> None:
        with self.voiceover(text=NARRATION['the_greedy_guess']):
            self.play(Indicate(self.markers[self.decisive_row], color=WHITE, scale_factor=1.25),
                      run_time=0.9)

        with self.voiceover(text=NARRATION['the_total']):
            self.total_readout = self._build_total_readout()
            self.play(FadeIn(self.total_readout), run_time=0.8)

    # ── Act five: what the solver actually does ───────────────────────────────────────

    def play_act_five_optimal_assignment(self) -> None:
        naive_column = self.naive_assignment[self.decisive_row]
        optimal_column = self.optimal_assignment[self.decisive_row]
        # Which future pick inherits the slot the drafted player gives up. Because the future rows
        # were aligned in setup, there is exactly one, and the whole difference between the two
        # assignments is the two-cycle between it and the decisive row.
        inheriting_row = next(
            row for row in range(self.drafted_count, self.slot_count)
            if self.optimal_assignment[row] == naive_column)

        # Every marker moves to the solved assignment at once, so the answer arrives as one
        # permutation rather than as a sequence of local repairs -- which is what the Hungarian
        # algorithm returns and, more to the point, is why it cannot be reasoned out row by row.
        with self.voiceover(text=NARRATION['solve_it_whole']):
            self.play(
                *[self.markers[row].animate.move_to(self._cell_centre(row, column))
                  for row, column in enumerate(self.optimal_assignment)],
                self.total_number.animate.set_value(self.prepared['optimal_total']),
                run_time=1.6,
            )

        with self.voiceover(text=NARRATION['the_swap']):
            self.play(
                Indicate(self.markers[self.decisive_row], color=GREEN_B, scale_factor=1.3),
                Indicate(self.markers[inheriting_row], color=GREEN_B, scale_factor=1.3),
                run_time=1.2,
            )

        # The close is about what the solved board is FOR: it says which slots the picks still
        # to come will land in, so lighting those rows up is the line's own subject. What is
        # missing from them is the point -- no future pick reaches a centre slot, because both
        # are already held, and that is a rebound and block total the team will not get.
        future_rows = range(self.drafted_count, self.slot_count)
        with self.voiceover(text=NARRATION['why_important']):
            self.play(*[Indicate(self.markers[row], color=BLUE_B, scale_factor=1.18)
                        for row in future_rows], run_time=1.6)
            closed_columns = sorted({self.optimal_assignment[row]
                                     for row in range(self.drafted_count)})
            self.play(*[Indicate(self.slot_headers[column], color=RED_D, scale_factor=1.2)
                        for column in closed_columns], run_time=1.4)
            self.wait(1.0)

    def construct(self) -> None:
        self.set_speech_service(GTTSService())
        self.build_grid_frame()
        self.play_act_one_eligibility()
        self.play_act_two_future_rewards()
        self.play_act_three_constraint()
        self.play_act_four_naive_assignment()
        self.play_act_five_optimal_assignment()
