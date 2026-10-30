"""The field works out what it is doing, and the algorithm watches it happen.

Every other scene in this set is about one team's decision. This one is about the fact that the
opposition is solving the same problem, and that the algorithm's picture of them is not assumed
but DISCOVERED -- by running H-scoring against itself until the answers stop moving.

Each row is one of the top players, each column a category, and the number is the win rate that
player's build buys in that category against the field: fifty percent is a coin flip, green is
above it, red below, on the app's own colour scale. Every player descends from every seed at once,
and a row shows the seed the field currently uses for that player. Two things are marked, every
iteration, because they are what keeps the field from lurching:

  - a player who SWITCHES seed -- at most six per iteration, the ones with the most to gain -- has
    their row outlined and their name highlighted, and their row can jump;
  - a 'top 12' tag marks the players the field is made of (its seats and spare); the field refreshes
    them every ten iterations, by the players' current H-scores, and the tags move with it.

The rows themselves never move: each player keeps one place, in the order of the final served
board, so a board that has settled is one whose colours have stopped changing.

The board opens full rather than empty: Level 0 has already solved everybody against a neutral
field before self-play starts, and that is what the opening frame shows.

Everything on screen is a recording of the shipped self-play, not a model of it -- the same
descent that runs at session build. Run `python visualizations/scenes/self_play/prepare_self_play_data.py`
first, which re-runs it with the recorders attached and writes what it saw.

    manim -ql visualizations/scenes/self_play/self_play.py SelfPlayLoop
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Rectangle, Text, DecimalNumber,
    FadeIn, FadeOut,
    RIGHT, LEFT, DOWN,
    WHITE, GREY_B, GREY_E, YELLOW,
    rgb_to_color,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.modify_audio import get_duration

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION


# ── Layout ───────────────────────────────────────────────────────────────────────────

CELL_WIDTH, CELL_HEIGHT = 0.80, 0.36
CELL_GAP_X, CELL_GAP_Y  = 0.85, 0.42
GRID_LEFT_X   = -4.10      # centre of the first column
GRID_TOP_Y    = 1.95       # centre of the first row
NAME_GAP      = 0.28       # between a player's name and his first cell
HEADER_LIFT   = 1.05       # how far above the first row the category names sit
OUTLINE_PAD   = 0.05       # how far a switched row's outline stands off its cells
TAG_GAP       = 0.12       # between a row's last cell and its 'new seed' tag
LEGEND_LEFT_X = 3.85       # the legend's left edge, clear of the tags and the top-twelve mark

# A switch is the event worth catching on a fast-moving board, so it gets the one strong colour.
SWITCH_COLOUR = YELLOW

# Cells are coloured by the app's own stat styler, so a viewer who has seen the H-score table
# is looking at the same scale here. Ported from frontend/styles/styler_functions.ts:
# darkPrimary(), driven by H_MULTIPLIER, which is what the app uses for per-category cells --
# a win percentage around a middle of fifty.
PARITY_WIN_RATE_PERCENT = 50
WIN_RATE_MULTIPLIER     = 3     # H_MULTIPLIER
INTENSITY_CAP           = 110   # where the app's ramp stops getting stronger

# The lines that play over the descent. Their measured lengths, divided by the number of recorded
# iterations, set ONE rate for the whole run: the board moves at a steady speed and finishes exactly
# when the narration does, instead of speeding up and slowing down beat by beat.
STEP_NARRATION_KEYS = ('early_steps', 're_estimation', 'seed_switching', 'late_steps')

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / 'prepared_data' / 'self_play.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/scenes/self_play/prepare_self_play_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def build_small_text(content: str, font_size: float, color) -> Text:
    """Text at a small size with its spaces intact: Pango drops narrow spaces when it lays text out
    small ('new seed' came out 'newseed'), so lay it out at four times the size and scale it down."""
    return Text(content, font_size=4 * font_size, color=color).scale(0.25)


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
    """The real self-play descent, iteration by iteration: each player's current seed's win rates, who switched
    seed, and who the field is made of."""

    def setup(self) -> None:
        self.measured = load_measurements()
        self.steps = self.measured['steps']
        self.categories = self.measured['categories']
        self.player_names = self.measured['player_names']
        self.opening_rates = np.array(self.measured['opening_rates'], dtype=float)
        self.context_size = self.measured['context_size']
        # Every player keeps one row, in the order of the final served board.
        self.slot_of_player = list(range(len(self.player_names)))

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
        """One outline per row, hidden until that player switches seed."""
        width = (len(self.categories) - 1) * CELL_GAP_X + CELL_WIDTH + 2 * OUTLINE_PAD
        return VGroup(*[
            Rectangle(width=width, height=CELL_HEIGHT + OUTLINE_PAD, stroke_color=SWITCH_COLOUR, stroke_width=2,
                      stroke_opacity=0.0, fill_opacity=0.0)
            .move_to(self.row_centre(self.slot_of_player[player_index]))
            for player_index in range(len(self.player_names))
        ])

    def tag_position(self, slot: int) -> np.ndarray:
        return self.cell_position(slot, len(self.categories) - 1) + np.array(
            [CELL_WIDTH / 2 + TAG_GAP, 0.0, 0.0])

    def build_field_tags(self, opening_members: list[bool]) -> VGroup:
        """One 'top 12' tag per row, shown while that player is one of the players the field is made of."""
        return VGroup(*[
            build_small_text(f'top {self.context_size}', 12, WHITE)
            .set_opacity(1.0 if member else 0.0)
            .move_to(self.tag_position(self.slot_of_player[player_index]), aligned_edge=LEFT)
            for player_index, member in enumerate(opening_members)
        ])

    def build_names(self) -> VGroup:
        return VGroup(*[
            Text(name, font_size=13, color=GREY_B)
            .move_to(self.name_position(self.slot_of_player[player_index]), aligned_edge=RIGHT)
            for player_index, name in enumerate(self.player_names)
        ])

    def build_headers(self) -> VGroup:
        return VGroup(*[
            Text(category.replace(' %', '%'), font_size=13, color=GREY_B)
            .rotate(np.pi / 2.6)
            .move_to(self.cell_position(0, category_index) + np.array([0.0, HEADER_LIFT, 0.0]))
            for category_index, category in enumerate(self.categories)
        ])

    def at_row(self, group: VGroup, player_index: int, category_index: int):
        return group[player_index * len(self.categories) + category_index]

    # ── The descent ───────────────────────────────────────────────────────────────────

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

    def play_steps(self, first: int, last: int, run_time: float) -> None:
        """Every recorded iteration in a range, each taking the same time as every other in the scene.

        The rows never move. The outlines and names show who switches seed, and the tags who the
        field is made of. Each mobject gets exactly ONE animate chain per step: two animations
        handed the same mobject in one play() would have the second discard the first.
        """
        for step in range(first, last):
            entry = self.steps[step]
            rates = np.array(entry['rates'], dtype=float)

            changes = [self.iteration_readout[1].animate.set_value(entry['iteration'])]
            for player_index in range(len(self.player_names)):
                switched = entry['switched'][player_index]
                name_colour = SWITCH_COLOUR if switched else GREY_B
                changes.append(self.names[player_index].animate.set_color(name_colour))
                changes.append(self.outlines[player_index].animate.set_stroke(opacity=1.0 if switched else 0.0))
                changes.append(self.field_tags[player_index].animate
                               .set_opacity(1.0 if entry['in_field'][player_index] else 0.0))
                for category_index in range(len(self.categories)):
                    win_rate = rates[player_index][category_index]
                    changes.append(
                        self.at_row(self.grid, player_index, category_index).animate
                        .set_fill(win_rate_colour(win_rate), opacity=1.0))
                    # The value is set OUTRIGHT, not animated. Animating it interpolates the
                    # glyph outlines, so a cell going from nine percent to fifteen spends the
                    # transition as a smear that reads as a rendering fault; the number simply
                    # is what it now is.
                    readout = self.at_row(self.readouts, player_index, category_index)
                    if np.isfinite(win_rate):
                        readout.set_value(round(win_rate * 100))
            self.play(*changes, run_time=run_time)

    # ── The scene ─────────────────────────────────────────────────────────────────────

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.grid = self.build_grid()
        self.readouts = self.build_readouts()
        self.names = self.build_names()
        self.outlines = self.build_row_outlines()
        self.field_tags = self.build_field_tags(self.measured['opening_in_field'])
        headers = self.build_headers()

        self.iteration_readout = VGroup(
            Text('iteration', font_size=17, color=GREY_B),
            DecimalNumber(0, num_decimal_places=0, font_size=28, color=WHITE),
        ).arrange(RIGHT, buff=0.24).move_to([5.35, 2.55, 0])
        legend = VGroup(
            build_small_text("current seed's win rate", 14, GREY_B),
            build_small_text(f'top {self.context_size}: in the field', 14, GREY_B),
            build_small_text('(refreshed every 10 iterations)', 14, GREY_B),
            build_small_text('switched seed', 14, SWITCH_COLOUR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to([LEGEND_LEFT_X, 1.05, 0], aligned_edge=LEFT)

        # The board goes up before a word is said about it: the opening line describes what is
        # on it, and describing a thing that is not there yet asks the viewer to wait for the
        # subject of the sentence.
        #
        # Nothing is pointed out on it either. Which categories the field gives up is legible
        # from the board itself -- whole columns of single digits -- and marking them told the
        # viewer something they had already read.
        self.add(self.outlines)
        self.play(FadeIn(self.grid), FadeIn(self.readouts), FadeIn(self.names),
                  FadeIn(headers), FadeIn(self.iteration_readout), FadeIn(legend),
                  FadeIn(self.field_tags), run_time=1.2)
        with self.voiceover(text=NARRATION['opening']):
            self.wait(1.0)

        # One rate for every step: the lines that narrate them, divided by the steps
        # there are to show. Each line then covers as many steps as fit at that rate, so the
        # board moves steadily and runs out exactly as the last line does.
        spoken = [self.narration_seconds(key) for key in STEP_NARRATION_KEYS]
        run_time = sum(spoken) / len(self.steps)
        played = 0
        for key, seconds in zip(STEP_NARRATION_KEYS, spoken):
            share = round(seconds / run_time)
            last = len(self.steps) if key == STEP_NARRATION_KEYS[-1] else played + share
            with self.voiceover(text=NARRATION[key]):
                self.play_steps(played, min(last, len(self.steps)), run_time)
            played = last

        # Hold on the final board, because where the field ends up is a claim about the whole
        # board at once rather than about any one row.
        with self.voiceover(text=NARRATION['settled']):
            self.play(*[outline.animate.set_stroke(opacity=0.0) for outline in self.outlines],
                      *[name.animate.set_color(GREY_B) for name in self.names], run_time=0.6)
            self.wait(2.4)
        self.play(FadeOut(self.grid), FadeOut(self.readouts), FadeOut(self.names),
                  FadeOut(headers), FadeOut(self.iteration_readout), FadeOut(self.outlines),
                  FadeOut(self.field_tags), FadeOut(legend), run_time=0.9)
