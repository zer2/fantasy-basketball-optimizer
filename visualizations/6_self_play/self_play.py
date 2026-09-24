"""The field works out what it is doing, and the algorithm watches it happen.

Every other scene in this set is about one team's decision. This one is about the fact that the
opposition is solving the same problem, and that the algorithm's picture of them is not assumed
but DISCOVERED -- by running H-scoring against itself until the answers stop moving.

Each row is one of the top players, each column a category, and the number is the win rate that
player's build buys in that category against the field: fifty percent is a coin flip, green is
above it, red below, on the app's own colour scale. Two things move every pass. The half of the
universe being re-solved is outlined while it is solved, and the rows RE-SORT, because the field
re-ranks itself off a merged score frame as it goes -- who the top players are is being estimated
at the same time as how they should be built.

The board opens full rather than empty: Level 0 has already solved everybody against a neutral
field before the first pass runs, and that is what the opening frame shows.

Everything on screen is a recording of the shipped bootstrap, not a model of it -- the same loop
that runs at session build. Run `python visualizations/6_self_play/prepare_self_play_data.py`
first, which re-runs it with the recorders attached and writes what it saw.

    manim -ql visualizations/6_self_play/self_play.py SelfPlayLoop
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Rectangle, Line, Text, DecimalNumber,
    FadeIn, FadeOut,
    RIGHT, LEFT, UP,
    YELLOW, WHITE, GREY_B, GREY_D, GREY_E,
    rgb_to_color,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.modify_audio import get_duration

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION


# ── Layout ───────────────────────────────────────────────────────────────────────────

CELL_WIDTH, CELL_HEIGHT = 0.80, 0.36
CELL_GAP_X, CELL_GAP_Y  = 0.85, 0.42
GRID_LEFT_X   = -4.10      # centre of the first column
GRID_TOP_Y    = 1.95       # centre of the first row
NAME_GAP      = 0.28       # between a player's name and his first cell
HEADER_LIFT   = 1.05       # how far above the first row the category names sit

# Cells are coloured by the app's own stat styler, so a viewer who has seen the H-score table
# is looking at the same scale here. Ported from frontend/styles/styler_functions.ts:
# darkPrimary(), driven by H_MULTIPLIER, which is what the app uses for per-category cells --
# a win percentage around a middle of fifty.
PARITY_WIN_RATE_PERCENT = 50
WIN_RATE_MULTIPLIER     = 3     # H_MULTIPLIER
INTENSITY_CAP           = 110   # where the app's ramp stops getting stronger

# The lines that play over the passes. Their measured lengths, divided by the number of passes,
# set ONE rate for the whole run: the board moves at a steady speed and finishes exactly when
# the narration does, instead of speeding up and slowing down beat by beat.
PASS_NARRATION_KEYS = ('early_passes', 're_estimation', 'late_passes')

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'self_play.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/6_self_play/prepare_self_play_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def win_rate_colour(win_rate: float):
    """The colour the app would give this cell.

    Red below a coin flip, green above, and -- the part a flat red-to-green ramp loses -- blue
    rising with the strength of either, so a category given up entirely is a different colour
    from one merely conceded, and a category won outright is different from one just ahead.
    """
    if not np.isfinite(win_rate):
        return GREY_E
    raw = (win_rate * 100 - PARITY_WIN_RATE_PERCENT) * WIN_RATE_MULTIPLIER
    intensity = min(round(abs(raw)), INTENSITY_CAP)
    red = 55 if raw > 0 else 55 + intensity
    green = 55 + intensity if raw > 0 else 55
    blue = 70 + round(intensity * 0.7)
    return rgb_to_color(np.array([red, green, blue]) / 255)


class SelfPlayLoop(VoiceoverScene):
    """Thirty-two passes of the real bootstrap: the win rates settling, and the order with them."""

    def setup(self) -> None:
        self.measured = load_measurements()
        self.passes = self.measured['passes']
        self.categories = self.measured['categories']
        self.player_names = self.measured['player_names']
        self.opening_rates = np.array(self.measured['opening_rates'], dtype=float)
        self.context_size = self.measured['context_size']
        # Which slot each player currently occupies. Everything on a row is positioned from
        # this, so a re-sort is a change to one list and a move for everything keyed off it.
        self.slot_of_player = [0] * len(self.player_names)
        self.place_players(self.measured['opening_order'])

    def place_players(self, order: list[int]) -> None:
        for slot, player_index in enumerate(order):
            self.slot_of_player[player_index] = slot

    # ── The grid ──────────────────────────────────────────────────────────────────────

    def cell_position(self, slot: int, category_index: int) -> np.ndarray:
        return np.array([
            GRID_LEFT_X + category_index * CELL_GAP_X,
            GRID_TOP_Y - slot * CELL_GAP_Y,
            0.0,
        ])

    def row_centre(self, slot: int) -> np.ndarray:
        return (self.cell_position(slot, 0)
                + self.cell_position(slot, len(self.categories) - 1)) / 2

    def name_position(self, slot: int) -> np.ndarray:
        return self.cell_position(slot, 0) + np.array(
            [-(NAME_GAP + CELL_WIDTH / 2), 0.0, 0.0])

    def build_grid(self) -> VGroup:
        cells = VGroup()
        for player_index in range(len(self.player_names)):
            slot = self.slot_of_player[player_index]
            for category_index in range(len(self.categories)):
                cells.add(Rectangle(
                    width=CELL_WIDTH, height=CELL_HEIGHT, stroke_width=0,
                    fill_color=win_rate_colour(self.opening_rates[player_index][category_index]),
                    fill_opacity=1.0,
                ).move_to(self.cell_position(slot, category_index)))
        return cells

    def build_readouts(self) -> VGroup:
        """The win rate written in every cell, as a whole percentage, the way the app writes it."""
        readouts = VGroup()
        for player_index in range(len(self.player_names)):
            slot = self.slot_of_player[player_index]
            for category_index in range(len(self.categories)):
                win_rate = self.opening_rates[player_index][category_index]
                readouts.add(
                    DecimalNumber(round(win_rate * 100), num_decimal_places=0, font_size=18,
                                  color=WHITE, unit=r'\%', unit_buff_per_font_unit=0.0)
                    .move_to(self.cell_position(slot, category_index)))
        return readouts

    def build_row_outlines(self) -> VGroup:
        """A frame around each row, shown only while that player is being re-solved."""
        width = (len(self.categories) - 1) * CELL_GAP_X + CELL_WIDTH + 0.10
        return VGroup(*[
            Rectangle(
                width=width, height=CELL_HEIGHT + 0.10,
                stroke_color=YELLOW, stroke_width=2.0, stroke_opacity=0.0,
                fill_opacity=0.0,
            ).move_to(self.row_centre(self.slot_of_player[player_index]))
            for player_index in range(len(self.player_names))
        ])

    def build_names(self) -> VGroup:
        return VGroup(*[
            Text(name, font_size=13, color=GREY_B)
            .move_to(self.name_position(self.slot_of_player[player_index]), aligned_edge=RIGHT)
            for player_index, name in enumerate(self.player_names)
        ])

    def build_context_marker(self) -> VGroup:
        """Where the drafting context stops, marked at the right edge of the board.

        The rows below the mark are in the universe the passes solve, but outside the seats the
        field is made of -- so a player crossing this line is a change in who the algorithm
        thinks it is drafting against, which is the thing the re-sorting is for. Drawn at a slot
        boundary rather than against a player: the line stays put and the players move through
        it.
        """
        boundary = (self.cell_position(self.context_size - 1, 0)[1]
                    + self.cell_position(self.context_size, 0)[1]) / 2
        right_edge = self.cell_position(0, len(self.categories) - 1)[0] + CELL_WIDTH / 2
        rule = Line([right_edge - 0.1, boundary, 0], [right_edge + 0.55, boundary, 0],
                    color=GREY_B, stroke_width=2)
        caption = (Text(f'top {self.context_size}', font_size=14, color=GREY_B)
                   .next_to(rule, RIGHT, buff=0.12))
        return VGroup(rule, caption)

    def build_headers(self) -> VGroup:
        return VGroup(*[
            Text(category.replace(' %', '%'), font_size=13, color=GREY_B)
            .rotate(np.pi / 2.6)
            .move_to(self.cell_position(0, category_index) + np.array([0.0, HEADER_LIFT, 0.0]))
            for category_index, category in enumerate(self.categories)
        ])

    def at_row(self, group: VGroup, player_index: int, category_index: int):
        return group[player_index * len(self.categories) + category_index]

    # ── The passes ────────────────────────────────────────────────────────────────────

    def add_narration_key(self, key_text) -> None:
        """Bring on the legend line for the row outlines, as the first of them appears."""
        self.play(FadeIn(key_text), run_time=0.5)

    def narration_seconds(self, key: str) -> float:
        """How long a line takes to say, before anything has been drawn.

        The audio is generated (or read from cache) up front so the whole run can be paced from
        it. Asking for it here costs nothing on a re-render: the service caches by text, and the
        voiceover blocks below ask for the very same files.
        """
        # generate_from_text rather than the wrapper around it: the wrapper appends a cache
        # entry every time it is called, so asking here and again inside the voiceover block
        # would write the same line into the cache twice.
        service = self.speech_service
        data = service.generate_from_text(' '.join(NARRATION[key].split()))
        return get_duration(Path(service.cache_dir) / data['original_audio'])

    def play_passes(self, first: int, last: int, run_time: float) -> None:
        """Every pass in a range, each taking the same time as every other pass in the scene.

        Each mobject gets exactly ONE animate chain, restyling and moving together: a cell that
        is both recoloured and re-sorted in the same pass would otherwise be handed to two
        animations at once, and the second would discard the first.
        """
        for step in range(first, last):
            entry = self.passes[step]
            rates = np.array(entry['rates'], dtype=float)
            solving = set(entry['solving'])
            self.place_players(entry['order'])

            changes = [self.pass_readout[1].animate.set_value(entry['pass_index'])]
            for player_index in range(len(self.player_names)):
                slot = self.slot_of_player[player_index]
                changes.append(
                    self.outlines[player_index].animate
                    .set_stroke(opacity=1.0 if player_index in solving else 0.0)
                    .move_to(self.row_centre(slot)))
                changes.append(
                    self.names[player_index].animate
                    .set_color(WHITE if player_index in solving else GREY_D)
                    .move_to(self.name_position(slot), aligned_edge=RIGHT))
                for category_index in range(len(self.categories)):
                    win_rate = rates[player_index][category_index]
                    position = self.cell_position(slot, category_index)
                    changes.append(
                        self.at_row(self.grid, player_index, category_index).animate
                        .set_fill(win_rate_colour(win_rate), opacity=1.0)
                        .move_to(position))
                    # The value is set OUTRIGHT, not animated. Animating it interpolates the
                    # glyph outlines, so a cell going from nine percent to fifteen spends the
                    # transition as a smear that reads as a rendering fault. Only the move is
                    # animated; the number simply is what it now is.
                    readout = self.at_row(self.readouts, player_index, category_index)
                    if np.isfinite(win_rate):
                        readout.set_value(round(win_rate * 100))
                    changes.append(readout.animate.move_to(position))
            self.play(*changes, run_time=run_time)

    # ── The scene ─────────────────────────────────────────────────────────────────────

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.grid = self.build_grid()
        self.readouts = self.build_readouts()
        self.outlines = self.build_row_outlines()
        self.names = self.build_names()
        headers = self.build_headers()

        self.pass_readout = VGroup(
            Text('pass', font_size=17, color=GREY_B),
            DecimalNumber(0, num_decimal_places=0, font_size=28, color=WHITE),
        ).arrange(RIGHT, buff=0.24).move_to([5.35, 2.55, 0])
        legend = VGroup(
            Text('category win rate', font_size=17, color=GREY_B),
            Text('rows ranked by H-score', font_size=17, color=GREY_B),
        ).arrange(UP, aligned_edge=LEFT, buff=0.24).move_to([5.35, 1.05, 0])
        # Held back until there is something outlined to explain: a key to a mark that is not
        # on screen yet is a question rather than an answer.
        outline_legend = Text('outlined: re-solved this pass', font_size=17, color=YELLOW)
        outline_legend.next_to(legend, UP, aligned_edge=LEFT, buff=0.24)

        # The board goes up before a word is said about it: the opening line describes what is
        # on it, and describing a thing that is not there yet asks the viewer to wait for the
        # subject of the sentence.
        #
        # Nothing is pointed out on it either. Which categories the field gives up is legible
        # from the board itself -- whole columns of single digits -- and marking them told the
        # viewer something they had already read.
        context_marker = self.build_context_marker()
        self.play(FadeIn(self.grid), FadeIn(self.readouts), FadeIn(self.names),
                  FadeIn(headers), FadeIn(self.pass_readout), FadeIn(legend),
                  FadeIn(context_marker), run_time=1.2)
        with self.voiceover(text=NARRATION['opening']):
            self.wait(1.0)
        self.add(self.outlines)
        self.add_narration_key(outline_legend)

        # One rate for every pass: the three lines that narrate them, divided by the passes
        # there are to show. Each line then covers as many passes as fit at that rate, so the
        # board moves steadily and runs out exactly as the last line does.
        spoken = [self.narration_seconds(key) for key in PASS_NARRATION_KEYS]
        run_time = sum(spoken) / len(self.passes)
        played = 0
        for key, seconds in zip(PASS_NARRATION_KEYS, spoken):
            share = round(seconds / run_time)
            last = len(self.passes) if key == PASS_NARRATION_KEYS[-1] else played + share
            with self.voiceover(text=NARRATION[key]):
                self.play_passes(played, min(last, len(self.passes)), run_time)
            played = last

        # The grid has stopped changing; hold on it, because "it settled" is a claim about the
        # whole board at once rather than about any one row.
        with self.voiceover(text=NARRATION['settled']):
            self.play(*[outline.animate.set_stroke(opacity=0.0) for outline in self.outlines],
                      *[name.animate.set_color(GREY_B) for name in self.names], run_time=0.6)
            self.wait(2.4)
        self.play(FadeOut(self.grid), FadeOut(self.readouts), FadeOut(self.names),
                  FadeOut(headers), FadeOut(self.outlines), FadeOut(self.pass_readout),
                  FadeOut(legend), FadeOut(outline_legend), FadeOut(context_marker),
                  run_time=0.9)
