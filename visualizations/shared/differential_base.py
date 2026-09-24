"""Shared apparatus for the two-teams-and-a-histogram scenes.

Both scenes deal two thirteen-player rosters out of the pool a standard league drafts, total
each side, and drop the difference into a histogram between them. They differ only in what a
player contributes: their season average (team_differential.py) or one real week they actually
played (weekly_differential.py). Everything else -- layout, dealing, the histogram, the four
acts -- lives here so the two cuts stay comparable by construction rather than by discipline.

A scene subclass sets the class attributes at the top of DifferentialSceneBase and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, Group, VGroup, ImageMobject, Rectangle, Circle, Line, Text, MathTex,
    ValueTracker,
    FadeIn, FadeOut, Create, Write, Transform, always_redraw,
    DOWN, RIGHT, UP,
    BLUE_D, BLUE_B, RED_D, RED_B, YELLOW, WHITE, GREY_B, GREY_E, BLACK,
)


# ── Where things sit (Manim's frame is 14.2 x 8 units, origin at the centre) ──────────

ROSTER_CENTRE_X       = 5.35    # distance of each roster's middle from the centre line
ROSTER_COLUMN_GAP     = 0.98    # horizontal gap between a roster's two columns
ROSTER_ROW_GAP        = 0.86    # vertical gap between roster rows
ROSTER_TOP_Y          = 2.45
HEADSHOT_HEIGHT       = 0.74

HISTOGRAM_WIDTH_BESIDE_ROSTERS = 7.4    # the centre column while both rosters are on screen
HISTOGRAM_WIDTH_ALONE          = 11.6   # once the rosters have gone and the chart owns the frame
HISTOGRAM_HEIGHT               = 3.5    # how tall the tallest bar is allowed to grow
HISTOGRAM_BASELINE_Y           = -2.45

# Montage pacing, in two gears. The point lands within a few hundred draws; everything after is
# confirmation, and ninety-five percent of the draws go by in a handful of seconds.
#
# Re-timed for a voice that reads about a fifth faster than the one these were set against.
# The montage is the longest stretch in the scene that no sentence is describing while it runs,
# so it is the stretch that decides whether the scene feels like it is waiting. Both gears keep
# their shape -- fast start, faster finish -- at roughly half the wall-clock.
MONTAGE_EARLY_SIMULATIONS    = 500
MONTAGE_EARLY_CHUNKS         = 7
MONTAGE_EARLY_FIRST_RUN_TIME = 0.52   # seconds per chunk at the start, easing down to...
MONTAGE_EARLY_LAST_RUN_TIME  = 0.24   # ...this by the end of the early gear
MONTAGE_LATE_CHUNKS          = 7
MONTAGE_LATE_DURATION        = 1.6    # seconds for every draw after the early gear, all told

# What the montage spends after its last draw lands: the rosters going and the chart taking
# their width, or -- for a scene that keeps its rosters -- just the arithmetic being cleared.
DISMISSAL_SECONDS    = 1.6
READOUT_FADE_SECONDS = 0.5

TEAM_COLOURS       = (BLUE_D, RED_D)
TEAM_LIGHT_COLOURS = (BLUE_B, RED_B)
# Numbered, not lettered. A bare capital letter in running text is read by gTTS as the article
# "a", and every spelling that forces the letter sound ("Ay", "Aye", "eigh") comes out as "eye"
# instead -- measured against a reference the voice cannot misread, a run of spelled-out letters.
# A digit has no such problem: "team 1 wins the week" and "team one wins the week" are the same
# audio to within a spectral distance of 0.003. The screen and the narration agree either way,
# so the tie is broken by what can be said.
TEAM_LABELS        = ('Team 1', 'Team 2')

_DATA_DIR     = Path(__file__).resolve().parent.parent / 'prepared_data'
_HEADSHOT_DIR = Path(__file__).resolve().parent.parent / 'prepared_assets' / 'headshots'


def load_prepared_data(data_path: Path) -> dict:
    """The pool and the precomputed draws written by prepare_season_data.py."""
    if not data_path.exists():
        raise FileNotFoundError(
            f'{data_path} is missing. Run `python visualizations/shared/prepare_season_data.py` '
            f'first -- the scenes deliberately do no data fetching of their own.')
    prepared = json.loads(data_path.read_text(encoding='utf-8'))
    prepared['rosters'] = np.array(prepared['simulation_rosters'])
    prepared['totals']  = np.array(prepared['simulation_totals'])
    return prepared


class DifferentialSceneBase(Scene):
    # How many decimals the totals are written to. A week of basketball is whole made baskets,
    # so the weekly scenes set this to zero; season averages are genuinely fractional and keep
    # a decimal place.
    decimal_places = 1

    """Two rosters, a histogram, and the four acts that drive them.

    Each act is a method, so a subclass can render one at a time while tuning it or all four in
    sequence for the finished video, off a single definition.
    """

    # ── What a subclass sets ──────────────────────────────────────────────────────────
    data_filename      = 'pool_2025_26.json'
    differential_limit = 90      # the axis runs from -limit to +limit
    bin_width          = 5
    axis_tick_step     = 30
    axis_caption       = 'Team 1 minus Team 2'
    spread_caption     = 'σ = {spread:.0f} points per game'
    # What to write against the two spread markers, low side then high, or None to leave them
    # bare. The caption above the curve names sigma while it is up, but it goes when the bars go
    # and the markers stay standing through the algebra -- so a scene that keeps them past that
    # point wants them saying what they are on their own.
    spread_marker_labels = None
    # How far the montage may be COMPRESSED to fit the line it plays under, as a fraction of its
    # natural pace. At 1.0 it is never sped up: a budget shorter than its own pace is ignored and
    # the act overruns, which shows as silence at the end of that line while the dealing finishes.
    # Below 1.0 it fits itself to the speech instead, down to this floor and no further, because
    # past some point the early draws stop being readable and the sync is not worth it.
    montage_minimum_stretch = 1.0
    # Whether the rosters leave once the dealing is finished, handing their width to the chart
    # for the curve and the algebra that follow.
    dismiss_rosters_after_montage = True

    # ── Setup ─────────────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.load_dataset(self.data_filename)

    def load_dataset(self, data_filename: str) -> None:
        """Point the scene at a prepared dataset, ready to deal from the first draw.

        Split out of setup so one scene can run TWO simulations in sequence: the G-score story
        replays a fixed matchup in real weeks and then lets the draft vary as well, and those two
        only compare because everything about them except the data is identical. Every derived
        quantity is rebuilt here, the draw counter included, so a second simulation starts from
        an empty histogram instead of inheriting the first one's.
        """
        self.prepared = load_prepared_data(_DATA_DIR / data_filename)
        self.team_size = self.prepared['team_size']
        self.differentials = self.prepared['totals'][:, 0] - self.prepared['totals'][:, 1]

        self.bin_edges = np.arange(
            -self.differential_limit, self.differential_limit + self.bin_width, self.bin_width)
        # Which bar each simulation's differential belongs to, computed once; -1 for the handful
        # that land beyond the axis. Those are left out of the histogram rather than clamped into
        # the end bars: clamping would heap every extreme onto one bin and invent a spike there,
        # which is precisely the artefact this chart has to be trustworthy about.
        bin_indices = np.digitize(self.differentials, self.bin_edges) - 1
        beyond_axis = (bin_indices < 0) | (bin_indices >= len(self.bin_edges) - 1)
        self.bin_index_of_simulation = np.where(beyond_axis, -1, bin_indices)

        self.simulations_shown = ValueTracker(0)
        self.axis_width = ValueTracker(HISTOGRAM_WIDTH_BESIDE_ROSTERS)
        self._headshot_cache: dict[int, ImageMobject] = {}
        self._roster_images = Group()

    # ── Static furniture ──────────────────────────────────────────────────────────────

    def build_static_frame(self) -> None:
        """Team titles, the running totals, and the histogram's axis."""
        self.team_name_labels = VGroup(*[
            Text(label, font_size=26, color=colour, weight='BOLD')
            .move_to([side * ROSTER_CENTRE_X, ROSTER_TOP_Y + 0.72, 0])
            for label, colour, side in zip(TEAM_LABELS, TEAM_COLOURS, (-1, 1))
        ])

        # Faded in as a snapshot, then handed over to an always_redraw copy: the axis has to be
        # able to widen later, which a static mobject could not do, but FadeIn on a redrawing
        # mobject fights its own updater.
        axis_snapshot = self._build_axis()
        self.play(FadeIn(self.team_name_labels), FadeIn(axis_snapshot), run_time=1.0)
        self.remove(axis_snapshot)
        self.axis = always_redraw(self._build_axis)
        self.add(self.axis)

    def _build_axis(self) -> VGroup:
        """Baseline, ticks and caption, drawn at the axis's current width."""
        half_width = self.axis_width.get_value() / 2
        axis = VGroup(Line([-half_width, HISTOGRAM_BASELINE_Y, 0],
                           [half_width, HISTOGRAM_BASELINE_Y, 0],
                           color=GREY_B, stroke_width=2))

        ticks = VGroup()
        for tick_value in range(-self.differential_limit,
                                self.differential_limit + 1,
                                self.axis_tick_step):
            x_position = self._x_of_differential(tick_value)
            tick = Line([x_position, HISTOGRAM_BASELINE_Y, 0],
                        [x_position, HISTOGRAM_BASELINE_Y - 0.11, 0],
                        color=GREY_B, stroke_width=2)
            label = Text(f'{tick_value:+d}' if tick_value else '0', font_size=18, color=GREY_B)
            label.next_to(tick, DOWN, buff=0.1)
            ticks.add(tick, label)

        axis.add(ticks, Text(self.axis_caption, font_size=19, color=GREY_B)
                 .next_to(ticks, DOWN, buff=0.22))
        return axis

    # ── Geometry helpers ──────────────────────────────────────────────────────────────

    def _x_of_differential(self, differential: float) -> float:
        """Horizontal position of a differential on the axis, at its current width."""
        return (differential / self.differential_limit) * (self.axis_width.get_value() / 2)

    def _roster_slot_position(self, side_sign: int, slot_index: int) -> np.ndarray:
        """Centre of one roster slot: two columns filling downward, thirteen slots per team."""
        column, row = slot_index % 2, slot_index // 2
        return np.array([
            side_sign * ROSTER_CENTRE_X + (column - 0.5) * ROSTER_COLUMN_GAP,
            ROSTER_TOP_Y - row * ROSTER_ROW_GAP,
            0.0,
        ])

    def _headshot(self, pool_index: int) -> ImageMobject:
        """A copy of one player's circular headshot, read from disk at most once per player."""
        if pool_index not in self._headshot_cache:
            player_id = self.prepared['pool'][pool_index]['player_id']
            image = ImageMobject(str(_HEADSHOT_DIR / f'{player_id}.png'))
            image.height = HEADSHOT_HEIGHT
            self._headshot_cache[pool_index] = image
        return self._headshot_cache[pool_index].copy()

    def _headshot_chip(self, pool_index: int, team_colour) -> Group:
        """One roster slot: the circular portrait on a disc, ringed in its team's colour.

        The ring is not decoration. NBA headshots are cut out against a dark background, so
        against this scene's black ground an unringed portrait has no edge and thirteen of them
        read as a smear rather than as a roster.
        """
        backing = Circle(
            radius       = HEADSHOT_HEIGHT / 2,
            fill_color   = GREY_E,
            fill_opacity = 1.0,
            stroke_color = team_colour,
            stroke_width = 2.5,
        )
        return Group(backing, self._headshot(pool_index))

    def _roster_images_for(self, simulation_index: int) -> Group:
        """Both rosters of one simulation, each headshot already in its slot."""
        drawn = self.prepared['rosters'][simulation_index]
        images = Group()
        for side_index, side_sign in enumerate((-1, 1)):
            offset = side_index * self.team_size
            for slot_index in range(self.team_size):
                chip = self._headshot_chip(drawn[offset + slot_index],
                                           TEAM_LIGHT_COLOURS[side_index])
                chip.move_to(self._roster_slot_position(side_sign, slot_index))
                images.add(chip)
        return images

    # ── The histogram ─────────────────────────────────────────────────────────────────

    def _counts_after(self, simulations_shown: float) -> np.ndarray:
        """Bar counts once `simulations_shown` draws have landed."""
        counts = np.zeros(len(self.bin_edges) - 1)
        completed = int(simulations_shown)
        if completed:
            drawn_bins = self.bin_index_of_simulation[:completed]
            landed, tallies = np.unique(drawn_bins[drawn_bins >= 0], return_counts=True)
            counts[landed] = tallies
        return counts

    def _height_scale(self, counts: np.ndarray) -> float:
        """Units of bar height per simulation, rescaled as the tallest bar grows.

        A fixed scale would make the first handful of draws invisible against ten thousand, so
        the axis grows with the data instead -- the shape is the subject here, not the counts.
        """
        return HISTOGRAM_HEIGHT / max(4.0, counts.max() * 1.12)

    def win_rate_for_left_team(self) -> float:
        """How often the left team came out ahead, across every simulation drawn so far."""
        shown = int(self.simulations_shown.get_value())
        drawn = self.differentials[:shown]
        return float((drawn > 0).mean()) if len(drawn) else 0.0

    def bars_above_zero(self) -> VGroup:
        """The bars on the winning side, as a group that can be coloured on its own.

        Rebuilt from the frozen bars rather than tracked as they are added: a bar's bin is what
        decides which side it is on, and the bin is recoverable from where it stands.
        """
        winning = VGroup()
        for bar in self.frozen_bars:
            if bar.get_center()[0] > self._x_of_differential(0.0):
                winning.add(bar)
        return winning

    def _build_bars(self) -> VGroup:
        """The histogram as it stands, one rectangle per non-empty bin."""
        counts = self._counts_after(self.simulations_shown.get_value())
        height_scale = self._height_scale(counts)
        bar_width = self.axis_width.get_value() / len(counts)

        bars = VGroup()
        for bin_index, count in enumerate(counts):
            if not count:
                continue
            bar_height = count * height_scale
            bar = Rectangle(
                width        = bar_width * 0.86,
                height       = bar_height,
                fill_color   = YELLOW,
                fill_opacity = 0.85,
                stroke_color = BLACK,
                stroke_width = 1,
            )
            bin_centre = self.bin_edges[bin_index] + self.bin_width / 2
            bar.move_to([self._x_of_differential(bin_centre),
                         HISTOGRAM_BASELINE_Y + bar_height / 2,
                         0])
            bars.add(bar)
        return bars

    def start_live_histogram(self) -> None:
        """Attach the histogram and the counter, both redrawing off the simulation tracker."""
        self.histogram = always_redraw(self._build_bars)
        self.simulation_counter = always_redraw(lambda: Text(
            f'{int(self.simulations_shown.get_value()):,} simulation'
            f'{"" if int(self.simulations_shown.get_value()) == 1 else "s"}',
            font_size=20, color=GREY_B,
        ).move_to([0, HISTOGRAM_BASELINE_Y - 1.15, 0]))
        self.add(self.histogram, self.simulation_counter)

    # ── Readouts ──────────────────────────────────────────────────────────────────────

    def _differential_readout(self, simulation_index: int) -> VGroup:
        """The one-line arithmetic above the histogram for a given simulation.

        Each total is coloured to the roster it came from, in the same shade that roster's
        headshots are ringed in. The totals used to be written a second time underneath the
        rosters, in a pair of numbers that arrived a beat BEFORE this line did and said exactly
        what it says -- so the colour does that job here, where the subtraction is, and the two
        numbers are attached to their teams without a second copy of them on the board.

        Built from separate pieces rather than as one string so the colours can be applied to
        the numbers and not to the arithmetic between them.
        """
        left_total, right_total = self.prepared['totals'][simulation_index]
        places = self.decimal_places
        readout = VGroup(
            Text(f'{left_total:.{places}f}', font_size=30, color=TEAM_LIGHT_COLOURS[0]),
            Text('-', font_size=30, color=GREY_B),
            Text(f'{right_total:.{places}f}', font_size=30, color=TEAM_LIGHT_COLOURS[1]),
            Text('=', font_size=30, color=GREY_B),
            Text(f'{left_total - right_total:+.{places}f}', font_size=30, color=WHITE),
        ).arrange(RIGHT, buff=0.26)
        readout.move_to([0, ROSTER_TOP_Y + 0.72, 0])
        return readout

    # ── Act one: one draw, slowly enough to read ──────────────────────────────────────

    def play_act_one_single_draw(self) -> None:
        images = self._roster_images_for(0)

        # Deal the two rosters a slot at a time. Nothing is counted while they land: a pair of
        # running totals used to climb under the rosters here, and they said the same thing the
        # subtraction above the histogram says a moment later, only sooner and twice.
        for slot_index in range(self.team_size):
            animations = [
                FadeIn(images[side_index * self.team_size + slot_index], scale=0.6)
                for side_index in range(2)
            ]
            self.play(*animations, run_time=0.34 if slot_index < 3 else 0.16)

        self._roster_images = images
        self.wait(0.6)

    def drop_first_bar(self) -> None:
        """The first bar falls out of the arithmetic: same number, now a position on the axis.

        Separate from dealing the teams because it is where the SIMULATION starts, and a scene
        may want to say so before it does. Must run before act two, which assumes a histogram
        to add to.

        The arithmetic arrives here too, rather than at the end of the dealing. Written there it
        sat at the top of the frame through the whole pause before the simulation began, posting
        a result for a histogram that did not exist yet -- and the number only means anything as
        the thing the first bar is a picture of.
        """
        self.differential_readout = self._differential_readout(0)
        self.start_live_histogram()
        self.play(Write(self.differential_readout), run_time=0.9)
        self.play(self.simulations_shown.animate.set_value(1), run_time=0.8)
        self.wait(0.8)

    # ── Act two: a few more draws, quicker each time ───────────────────────────────────

    def play_act_two_repeated_draws(self, through_simulation: int = 6) -> None:
        for simulation_index in range(1, through_simulation):
            pace = max(0.28, 0.75 - 0.12 * simulation_index)
            replacement = self._roster_images_for(simulation_index)
            self.play(
                FadeOut(self._roster_images, scale=0.85),
                FadeIn(replacement, scale=1.15),
                run_time=pace,
            )
            self._roster_images = replacement

            new_readout = self._differential_readout(simulation_index)
            self.play(Transform(self.differential_readout, new_readout), run_time=pace * 0.6)
            self.play(self.simulations_shown.animate.set_value(simulation_index + 1),
                      run_time=pace * 0.7)
        self.wait(0.4)

    # ── Act three: the montage ────────────────────────────────────────────────────────

    def play_act_three_montage(
        self
        , through_simulation: int | None = None
        , seconds_available: float | None = None
        , finish_the_frame: bool = True
    ) -> None:
        """Fill the histogram out to every draw the prepared data holds.

        The rosters keep dealing but stop being animated one headshot at a time: past a few draws
        the faces are no longer the point, and animating twenty-six images per draw is what would
        make this act cost more to render than the rest of the scene combined.

        `seconds_available` stretches the whole act to fill a stated time, for a scene whose line
        names the moment the dealing should end -- "the result is another bell curve" wants the
        dealing finished and the rosters gone as it is said, not half a sentence earlier. The
        two gears keep their shape and their ratio; only the clock changes.

        It never speeds the montage UP. The pace it runs at otherwise is the fastest the early
        draws stay readable at, so a budget shorter than that is taken as no budget at all and
        the act simply overruns, which is visible, rather than blurring, which is not.

        `finish_the_frame` is what clears the rosters away at the end. A scene that wants the
        dealing to run under one line and the rosters to leave under the NEXT one turns it off
        here and clears them itself, which is the only way to put a line break in the middle of
        a montage: an animation cannot span two voiceover blocks, but two montages can.
        """
        # Defaulting to the data's own length rather than a literal: the count lives in one place,
        # so raising it in prepare_season_data.py cannot leave the scene playing a fraction of the
        # draws while its counter claims otherwise.
        if through_simulation is None:
            through_simulation = len(self.differentials)
        start = int(self.simulations_shown.get_value())

        schedule = self._montage_schedule(start, through_simulation)
        if seconds_available is not None:
            # What the act spends outside the drawing: the beat after the last draw lands, and
            # the rosters leaving. Both are fixed, so only the draws take up the slack.
            fixed = 0.3
            if finish_the_frame:
                fixed += (DISMISSAL_SECONDS if self.dismiss_rosters_after_montage
                          else READOUT_FADE_SECONDS)
            drawing = sum(run_time for _, run_time in schedule)
            stretch = (max(self.montage_minimum_stretch, (seconds_available - fixed) / drawing)
                       if drawing else 1.0)
            schedule = [(checkpoint, run_time * stretch) for checkpoint, run_time in schedule]

        for checkpoint, run_time in schedule:
            simulation_index = checkpoint - 1

            # An instant swap, not an animated one: this reads as dealing at speed, and costs
            # nothing per frame.
            self.remove(self._roster_images)
            self._roster_images = self._roster_images_for(simulation_index)
            self.add(self._roster_images)
            self.differential_readout.become(self._differential_readout(simulation_index))

            self.play(self.simulations_shown.animate.set_value(checkpoint), run_time=run_time)

        self.wait(0.3)
        if not finish_the_frame:
            return
        if self.dismiss_rosters_after_montage:
            self.dismiss_rosters()
        else:
            self.play(FadeOut(self.differential_readout), run_time=READOUT_FADE_SECONDS)

    def dismiss_rosters(self) -> None:
        """Clear the rosters and give the freed width to the histogram.

        Only once the dealing is over. The rosters are what the histogram is a record OF, so
        taking them away while draws are still landing breaks the link the scene spent its first
        twenty seconds establishing. Afterwards they have nothing left to say, and the curve is
        worth the two thirds of the frame they were occupying.
        """
        self.play(
            FadeOut(self._roster_images, scale=0.8),
            FadeOut(self.team_name_labels),
            FadeOut(self.differential_readout),
            run_time=DISMISSAL_SECONDS * 0.44,
        )
        self._roster_images = Group()
        self.play(self.axis_width.animate.set_value(HISTOGRAM_WIDTH_ALONE),
                  run_time=DISMISSAL_SECONDS * 0.56)

    def _montage_schedule(self, start: int, through_simulation: int) -> list[tuple[int, float]]:
        """[(simulations completed, seconds to get there)] for the montage, in two gears.

        The first few hundred draws are the ones that teach: bins are empty, the outline is
        arriving, and a viewer is still learning to read the chart. Those play slowly enough to
        follow. By five hundred the point has landed and every further draw only firms up a shape
        already visible, so the remaining ninety-five percent of them go past in a few seconds --
        which is also the honest picture of what ten thousand draws is, next to five hundred.

        The late gear is set by how long it should take rather than as a multiple of the early
        rate: "flies by" is a statement about seconds on screen, and pinning the duration keeps it
        true no matter how the early gear is retuned.
        """
        early_end = min(MONTAGE_EARLY_SIMULATIONS, through_simulation)
        early_checkpoints = np.linspace(start, early_end, MONTAGE_EARLY_CHUNKS + 1)[1:]
        early_run_times = np.linspace(
            MONTAGE_EARLY_FIRST_RUN_TIME, MONTAGE_EARLY_LAST_RUN_TIME, MONTAGE_EARLY_CHUNKS)

        schedule = [(int(checkpoint), float(run_time))
                    for checkpoint, run_time in zip(early_checkpoints, early_run_times)]
        if through_simulation <= early_end:
            return schedule

        late_checkpoints = np.linspace(early_end, through_simulation, MONTAGE_LATE_CHUNKS + 1)[1:]
        late_run_time = MONTAGE_LATE_DURATION / MONTAGE_LATE_CHUNKS
        schedule += [(int(checkpoint), late_run_time) for checkpoint in late_checkpoints]
        return schedule

    # ── Act four: the shape it was always going to be ─────────────────────────────────

    def curve_height_at(self, differential: float) -> float:
        """The fitted curve at a differential, in the bar-height units it is drawn in.

        Only meaningful once act four has fitted it, which is also the only point at which
        anything has a curve to ask about.
        """
        density = np.exp(-0.5 * ((differential - self.curve_mean) / self.curve_spread) ** 2) / (
            self.curve_spread * np.sqrt(2 * np.pi))
        return self.curve_simulations * self.bin_width * density * self.curve_height_scale

    def play_act_four_normal_curve(self) -> None:
        completed = int(self.simulations_shown.get_value())
        sample = self.differentials[:completed]
        mean, standard_deviation = sample.mean(), sample.std()

        # Freeze the histogram: the curve is a claim about the bars as they stand, and bars that
        # kept redrawing underneath it would make that comparison meaningless.
        counts = self._counts_after(completed)
        height_scale = self._height_scale(counts)
        self.frozen_bars = self._build_bars()
        self.remove(self.histogram)
        self.add(self.frozen_bars)

        # Kept on the scene rather than closed over here: what the curve is worth at a given
        # differential is the quantity the rest of the scene is about, and a scene that wants to
        # shade a slice of it should read the same curve rather than fit its own.
        self.curve_mean = mean
        self.curve_spread = standard_deviation
        self.curve_height_scale = height_scale
        self.curve_simulations = completed

        curve_points = [
            [self._x_of_differential(differential),
             HISTOGRAM_BASELINE_Y + self.curve_height_at(differential),
             0]
            for differential in np.linspace(
                -self.differential_limit, self.differential_limit, 240)
        ]
        self.normal_curve = VGroup(*[
            Line(start, end, color=WHITE, stroke_width=4)
            for start, end in zip(curve_points, curve_points[1:])
        ])
        self.curve_peak_y = HISTOGRAM_BASELINE_Y + self.curve_height_at(mean)
        self.play(Create(self.normal_curve), run_time=1.6)

        self.spread_label = Text(
            self.spread_caption.format(spread=standard_deviation), font_size=22, color=WHITE,
        ).move_to([0, ROSTER_TOP_Y + 0.72, 0])
        self.play(FadeIn(self.spread_label, shift=DOWN * 0.2), run_time=0.8)

        # One standard deviation either side of dead level, marked on the axis: the width of a
        # typical mismatch is the quantity the rest of the algorithm is built around.
        self.spread_markers = VGroup(*[
            Line([self._x_of_differential(mean + sign * standard_deviation),
                  HISTOGRAM_BASELINE_Y, 0],
                 [self._x_of_differential(mean + sign * standard_deviation),
                  HISTOGRAM_BASELINE_Y + HISTOGRAM_HEIGHT * 0.72, 0],
                 color=GREY_B, stroke_width=3)
            for sign in (-1, 1)
        ])
        self.play(Create(self.spread_markers), run_time=0.8)

        # Written at the top of each marker rather than under the axis, where the tick numbers
        # already are: two more labels down there would read as more of the same scale instead
        # of as a name for the line above them.
        if self.spread_marker_labels is not None:
            self.spread_marker_names = VGroup(*[
                Text(label, font_size=24, color=GREY_B).next_to(marker, UP, buff=0.12)
                for label, marker in zip(self.spread_marker_labels, self.spread_markers)
            ])
            self.play(FadeIn(self.spread_marker_names), run_time=0.5)
        self.wait(1.2)

    # ── Act five: the height of the curve at nothing-to-choose-between-them ───────────

    def clear_for_the_formula(self) -> None:
        """Take the bars away, leaving the curve the algebra is about.

        Split out of act five so a scene can put a line between this and the formula. The
        sentence that introduces the formula arrives several seconds after the one about the
        height in the middle, and writing it the moment the bars left showed the answer while
        the question was still being asked.

        The spread markers stay. They are the width the formula is about, so keeping them
        standing while the bars go leaves sigma visible on the picture the algebra describes.
        """
        self.play(
            FadeOut(self.frozen_bars),
            FadeOut(self.spread_label),
            FadeOut(self.simulation_counter),
            run_time=0.9,
        )

    def play_act_five_density_at_zero(self) -> None:
        """Clear everything but the curve, and read its peak off the formula.

        The whole apparatus existed to establish one quantity: how much probability sits at a
        dead heat. That is the density at zero, and for a Normal it collapses to a single term
        -- the exponential vanishes and the height is one over sigma root two pi. It is why
        the spread is the number worth knowing: the tighter the distribution, the more a small
        edge moves the odds.
        """
        self.wait(0.3)

        # Written with mu already zero rather than carried through and cancelled, because both
        # teams are dealt from the same pool and the curve is centred on a dead heat. Starting
        # there means setting x to zero collapses the whole exponential in one step, which is
        # the move worth watching.
        equation = MathTex(
            r'f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{x^2}{2\sigma^2}}',
            font_size=46, color=WHITE,
        ).move_to([0, 2.58, 0])
        self.play(Write(equation), run_time=1.5)
        self.wait(1.2)

        # Asking for x = 0 is the whole question, so everything happens at once here: the
        # expression turns yellow and the height it is about rises on the curve. Colour is
        # doing the work a second copy of the formula would otherwise do -- the yellow line and
        # the yellow algebra are the same claim, and nothing needs to say so twice.
        self.peak_marker = Line([0, HISTOGRAM_BASELINE_Y, 0], [0, self.curve_peak_y, 0],
                                color=YELLOW, stroke_width=4)
        substituted = MathTex(
            r'f(0) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{0^2}{2\sigma^2}}',
            font_size=46, color=YELLOW,
        ).move_to(equation)
        self.play(Transform(equation, substituted), Create(self.peak_marker),
                  run_time=1.2)
        self.wait(1.2)

        collapsed = MathTex(
            r'f(0) = \frac{1}{\sigma\sqrt{2\pi}}',
            font_size=46, color=YELLOW,
        ).move_to(equation)
        self.play(Transform(equation, collapsed), run_time=1.1)
        # Kept so a scene can carry on from it. The Z-score scene takes this expression apart
        # again to reach the formula; the others end here.
        self.density_equation = equation
        # Short, because act six opens by fading all of this out. Two seconds of holding a
        # frame that is about to be cleared is two seconds the narration has to cover.
        self.wait(0.8)
