"""Why a percentage category is not an average of its players' percentages.

A team's field goal percentage is its made shots over its attempts. Written out, that is

    team% = sum(attempts_i * percent_i) / sum(attempts_i)

which is an average of the players' percentages WEIGHTED BY THEIR ATTEMPTS, not a plain one. The
two differ by a lot: on the draft this scene shows, one team's players average .506 between them
while the team shoots .486, because the players shooting best are the ones shooting least.

Rearranged around the league average it says something sharper still. Since the weights sum to
one,

    team% - league% = sum( (attempts_i / team attempts) * (percent_i - league%) )

so each player pulls the team off the league average by exactly their SHARE OF THE ATTEMPTS
times HOW FAR OFF THEY SHOOT, and the thirteen pulls add up to the whole difference with nothing
left over. That product is the quantity this act exists to show, and it is the same one the app
computes in backend/math/process_player_data.py as `ratio_num`.

Everything drawn is real: the pool is the 156 players a standard league drafts and the two teams
are the first draw of the same simulation the rest of the scene runs on, so these are the teams
the viewer has already watched being dealt.

A note on scale, because it decides what the act can show. A tank's height is a percentage from
nothing to everything, which is the honest way to draw a percentage and makes the players'
spread -- .403 to .726 here -- plainly visible. It also makes the two TEAMS look identical,
because .478 and .486 are eight thousandths apart and that is a pixel. That is not a drawing
problem to be solved by stretching the axis; it is the fact that thirteen players average out.
So the pulls are drawn as their own bars rather than as movements of the tank.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Rectangle, Circle, Line, DashedLine, Polygon, Text, MathTex,
    SurroundingRectangle,
    FadeIn, Create, Transform, Write,
    DOWN, UP, LEFT, RIGHT,
    BLUE_B, RED_B, WHITE, GREY_B, GREY_D, BLACK,
)

MADE_KEY = 'Field Goals Made'
ATTEMPTS_KEY = 'Field Goal Attempts'

# ── Where the two tanks stand ─────────────────────────────────────────────────────────
TANK_CENTRE_X = 3.55          # distance of each tank's middle from the centre line
TANK_BASELINE_Y = -2.95
TANK_HEIGHT = 2.50            # the whole height is a hundred percent
UNITS_PER_ATTEMPT = 0.0248    # a tank's width is the attempts its team takes

# ── The players standing over their team ──────────────────────────────────────────────
# Area, not radius, carries the attempts: a circle twice as wide reads as four times as much,
# and it is the reading that has to be true.
SQUARE_UNITS_PER_ATTEMPT = 0.0139
CIRCLE_ROW_Y = (0.65, 1.85)   # two rows, the taller circles of each row centred on it
CIRCLE_COLUMN_GAP = 0.74
CIRCLES_PER_ROW = 7

# Once the full-scale view has done its job, the tanks stretch to a band this far either side
# of the league percentage. Eight thousandths separate the two teams, which is a pixel across a
# tank that runs from nothing to everything; across a band of eight HUNDREDTHS it is a tenth of
# the tank. The band is marked at both ends, because a stretched axis that does not say so is
# just a lie about the size of a difference.
ZOOM_HALF_BAND = 0.04

# The two players the impact is shown on, the one taking fewer shots first. Chosen because they
# make the point against the grain: Robinson shoots .726 to Duren's .653, so he is the better
# shooter by far -- and moves the team less than half as much, because he takes a third of the
# shots. A pair where the better shooter also mattered more would illustrate nothing.
CONTRAST_PLAYERS = ('Mitchell Robinson', 'Jalen Duren')
CONTRAST_TEAM = 0
PULL_BAR_WIDTH = 0.34
DIMMED_OPACITY = 0.18
CONTRAST_PULSE = 1.18         # how far the two circles swell when the volume factor is named

# Above the circles, in the band of frame nothing else uses -- and offset to the right, because
# the left team's own captions reach up into it. Centred, the box drawn round the volume factor
# landed on the "2.3% of the shots" label sitting over the smallest circle.
VOLUME_FORMULA_CENTRE = [1.85, 3.05, 0.0]

LIQUID_OPACITY = 0.85
_DATA_PATH = Path(__file__).resolve().parent.parent / 'prepared_data' / 'pool_2025_26.json'


def load_shooting() -> dict:
    """The pool's shooting, the two teams drawn from it, and the league percentage.

    Read from the same file the histogram scene deals out of, so the two teams here are the
    two teams there -- the first draw of the same simulation, faces the viewer has seen.
    """
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run `python visualizations/shared/prepare_season_data.py` '
            f'-- the scenes deliberately do no data fetching of their own.')
    prepared = json.loads(_DATA_PATH.read_text(encoding='utf-8'))
    pool = prepared['pool']

    missing = [key for key in (MADE_KEY, ATTEMPTS_KEY) if key not in pool[0]]
    if missing:
        raise KeyError(
            f'the prepared pool has no {missing}. The shooting act needs made shots and '
            f'attempts per player; re-run prepare_season_data.py, which now carries them.')

    made = np.array([player[MADE_KEY] for player in pool])
    attempts = np.array([player[ATTEMPTS_KEY] for player in pool])
    team_size = prepared['team_size']
    roster = np.array(prepared['simulation_rosters'][0])[:2 * team_size]

    return {
        'names':          [player['name'] for player in pool],
        'made':           made,
        'attempts':       attempts,
        'team_size':      team_size,
        'teams':          [roster[:team_size], roster[team_size:]],
        'league_percent': float(made.sum() / attempts.sum()),
    }


class ShootingPercentageAct:
    """The tanks, the players over them, and the pull each player has on its level."""

    def setup_shooting(self) -> None:
        self.shooting = load_shooting()
        # The tanks start at full scale: a percentage drawn from nothing to everything, which is
        # what a percentage means before anything is done to it.
        self.zoomed = False

    # ── The quantities ────────────────────────────────────────────────────────────────

    def team_attempts(self, side: int) -> float:
        return float(self.shooting['attempts'][self.shooting['teams'][side]].sum())

    def team_percent(self, side: int) -> float:
        """Made over attempted -- which is NOT the average of the players' percentages."""
        roster = self.shooting['teams'][side]
        return float(self.shooting['made'][roster].sum()
                     / self.shooting['attempts'][roster].sum())

    def player_percents(self, side: int) -> np.ndarray:
        roster = self.shooting['teams'][side]
        return self.shooting['made'][roster] / self.shooting['attempts'][roster]

    def player_pulls(self, side: int) -> np.ndarray:
        """Each player's share of the attempts times how far off the league they shoot.

        These sum to exactly the team's distance from the league percentage, which is the
        identity the act is about and is worth asserting rather than trusting.
        """
        roster = self.shooting['teams'][side]
        attempts = self.shooting['attempts'][roster]
        shares = attempts / attempts.sum()
        deviations = self.player_percents(side) - self.shooting['league_percent']
        pulls = shares * deviations

        stated = self.team_percent(side) - self.shooting['league_percent']
        if abs(pulls.sum() - stated) > 1e-9:
            raise ValueError(
                f'the pulls sum to {pulls.sum():.6f} but the team stands {stated:.6f} off the '
                f'league percentage. They are the same quantity, so they cannot disagree.')
        return pulls

    # ── The drawing ───────────────────────────────────────────────────────────────────

    def _tank_width(self, side: int) -> float:
        return self.team_attempts(side) * UNITS_PER_ATTEMPT

    def _tank_centre_x(self, side: int) -> float:
        return (-1 if side == 0 else 1) * TANK_CENTRE_X

    def zoom_band(self) -> tuple[float, float]:
        """The percentages the stretched tank runs between, lowest first."""
        league = self.shooting['league_percent']
        return league - ZOOM_HALF_BAND, league + ZOOM_HALF_BAND

    def _height_of_percent(self, percent: float) -> float:
        """A percentage as a height above the tank floor.

        Full scale until the act stretches it: nothing at zero, the whole tank at one. Stretched,
        the tank floor and ceiling are the ends of a band around the league percentage, and both
        ends carry their number so the stretch is stated rather than hidden.
        """
        if not self.zoomed:
            return TANK_HEIGHT * percent
        floor, ceiling = self.zoom_band()
        return TANK_HEIGHT * (percent - floor) / (ceiling - floor)

    def _build_tank(self, side: int, colour) -> VGroup:
        """One team: an empty vessel as wide as its attempts, filled to its percentage."""
        width = self._tank_width(side)
        centre_x = self._tank_centre_x(side)

        vessel = Rectangle(
            width=width, height=TANK_HEIGHT,
            stroke_color=GREY_B, stroke_width=2, fill_opacity=0.0,
        ).move_to([centre_x, TANK_BASELINE_Y + TANK_HEIGHT / 2, 0])

        filled = self._height_of_percent(self.team_percent(side))
        liquid = Rectangle(
            width=width, height=filled,
            stroke_width=0, fill_color=colour, fill_opacity=LIQUID_OPACITY,
        ).move_to([centre_x, TANK_BASELINE_Y + filled / 2, 0])

        return VGroup(vessel, liquid)

    def _circle_radius(self, attempts: float) -> float:
        return math.sqrt(SQUARE_UNITS_PER_ATTEMPT * attempts / math.pi)

    def _circle_liquid(self, centre: np.ndarray, radius: float, percent: float) -> Polygon:
        """The part of a disc below its level, as the disc's own shape rather than a box."""
        floor_y = centre[1] - radius
        level_y = floor_y + 2 * radius * percent
        steps = 26

        left_edge, right_edge = [], []
        for step in range(steps + 1):
            y = floor_y + (level_y - floor_y) * step / steps
            half_width = math.sqrt(max(0.0, radius ** 2 - (y - centre[1]) ** 2))
            left_edge.append([centre[0] - half_width, y, 0.0])
            right_edge.append([centre[0] + half_width, y, 0.0])

        return Polygon(*left_edge, *reversed(right_edge),
                       stroke_width=0, fill_opacity=LIQUID_OPACITY)

    def _player_position(self, side: int, slot: int) -> np.ndarray:
        """Where one player's circle stands, in two rows over its own team's tank."""
        row, column = divmod(slot, CIRCLES_PER_ROW)
        in_row = min(CIRCLES_PER_ROW, self.shooting['team_size'] - row * CIRCLES_PER_ROW)
        return np.array([
            self._tank_centre_x(side) + (column - (in_row - 1) / 2) * CIRCLE_COLUMN_GAP,
            CIRCLE_ROW_Y[row],
            0.0,
        ])

    def _build_players(self, side: int, colour) -> VGroup:
        """One circle per player: as big as their attempts, as full as they shoot."""
        roster = self.shooting['teams'][side]
        attempts = self.shooting['attempts'][roster]
        percents = self.player_percents(side)

        # Biggest first, so the row reads as an ordering and the big shooters sit together.
        order = np.argsort(-attempts)
        players = VGroup()
        for slot, index in enumerate(order):
            centre = self._player_position(side, slot)
            radius = self._circle_radius(float(attempts[index]))
            outline = Circle(radius=radius, stroke_color=GREY_B, stroke_width=1.5,
                             fill_color=BLACK, fill_opacity=1.0).move_to(centre)
            liquid = self._circle_liquid(centre, radius, float(percents[index]))
            liquid.set_fill(colour, opacity=LIQUID_OPACITY)
            players.add(VGroup(outline, liquid))
        return players

    def _build_league_line(self) -> VGroup:
        """One dashed line across both tanks, at the percentage the whole pool shoots."""
        height = TANK_BASELINE_Y + self._height_of_percent(self.shooting['league_percent'])
        left = self._tank_centre_x(0) - self._tank_width(0) / 2 - 0.35
        right = self._tank_centre_x(1) + self._tank_width(1) / 2 + 0.35
        line = DashedLine([left, height, 0], [right, height, 0],
                          color=GREY_D, stroke_width=2, dash_length=0.12)
        label = Text(f'league {self.shooting["league_percent"]:.3f}'.replace('0.', '.'),
                     font_size=15, color=GREY_D)
        label.next_to(line, RIGHT, buff=0.12)
        return VGroup(line, label)

    def build_shooting_frame(self) -> VGroup:
        """Everything the act stands on, before anything is said about it."""
        colours = (BLUE_B, RED_B)
        self.tanks = VGroup(*[self._build_tank(side, colours[side]) for side in range(2)])
        self.player_circles = VGroup(*[self._build_players(side, colours[side])
                                       for side in range(2)])
        self.league_line = self._build_league_line()

        self.tank_labels = VGroup(*[
            Text(f'Team {side + 1}  {self.team_percent(side):.3f}'.replace('0.', '.'),
                 font_size=20, color=colours[side])
            .next_to(self.tanks[side], DOWN, buff=0.18)
            for side in range(2)
        ])
        self.attempts_labels = VGroup(*[
            Text(f'{self.team_attempts(side):.0f} attempts', font_size=15, color=GREY_B)
            .next_to(self.tank_labels[side], DOWN, buff=0.08)
            for side in range(2)
        ])

        return VGroup(self.tanks, self.player_circles, self.league_line,
                      self.tank_labels, self.attempts_labels)

    def _build_band_marks(self) -> VGroup:
        """The two numbers the stretched tanks run between, written on their outer edges."""
        floor, ceiling = self.zoom_band()
        marks = VGroup()
        for side in range(2):
            outward = LEFT if side == 0 else RIGHT
            edge_x = (self._tank_centre_x(side)
                      + (-1 if side == 0 else 1) * self._tank_width(side) / 2)
            for percent, height in ((ceiling, TANK_HEIGHT), (floor, 0.0)):
                tick = Line([edge_x, TANK_BASELINE_Y + height, 0],
                            [edge_x + (-0.16 if side == 0 else 0.16),
                             TANK_BASELINE_Y + height, 0],
                            color=GREY_B, stroke_width=2)
                label = Text(f'{percent:.3f}'.replace('0.', '.'), font_size=15, color=GREY_B)
                label.next_to(tick, outward, buff=0.08)
                marks.add(tick, label)
        return marks

    def play_zoom_tanks(self) -> None:
        """Stretch both tanks onto a band around the league percentage, and say which band.

        The full-scale view is the honest one and has to come first, because it is the one that
        says what a percentage is. It is also the one in which the two teams are indistinguishable
        -- eight thousandths apart on a tank that runs to one -- and the act cannot go on to talk
        about what moves that level while the level looks like it cannot move at all.
        """
        colours = (BLUE_B, RED_B)
        self.zoomed = True

        stretched = [self._build_tank(side, colours[side])[1] for side in range(2)]
        stretched_line = self._build_league_line()
        self.band_marks = self._build_band_marks()

        self.play(
            *[Transform(self.tanks[side][1], stretched[side]) for side in range(2)],
            Transform(self.league_line, stretched_line),
            run_time=1.3,
        )
        self.play(FadeIn(self.band_marks), run_time=0.6)
        self.wait(0.5)

    def _slot_of(self, side: int, name: str) -> int:
        """Where a named player's circle stands, in the order the circles were laid out."""
        roster = self.shooting['teams'][side]
        names = [self.shooting['names'][int(index)] for index in roster]
        if name not in names:
            raise ValueError(
                f'{name!r} is not on team {side + 1}. The contrast players are named rather '
                f'than picked by rule, so a change to the draw has to be reflected here.')
        seat = names.index(name)
        order = list(np.argsort(-self.shooting['attempts'][roster]))
        return order.index(seat)

    def _pull_of(self, side: int, name: str) -> tuple[float, float, float]:
        """(share of the team's attempts, distance from the league, their product)."""
        roster = self.shooting['teams'][side]
        names = [self.shooting['names'][int(index)] for index in roster]
        seat = names.index(name)
        attempts = self.shooting['attempts'][roster]
        share = float(attempts[seat] / attempts.sum())
        deviation = float(self.player_percents(side)[seat] - self.shooting['league_percent'])
        return share, deviation, share * deviation

    def play_pull_comparison(self) -> None:
        """What one player does to the level: their share of the shots times how far off they are.

        Both halves of the product are on screen already -- the size of a circle is the share and
        its distance from the league line is the deviation -- so the beat does not introduce
        anything, it just takes two of them and draws the product they come to.

        The pulls are thousandths, which is why this waits for the stretched tank. At full scale
        they would be a pixel, and the claim would have to be taken on trust.
        """
        if not self.zoomed:
            raise ValueError(
                'the pull bars are drawn against the stretched band and are a pixel tall '
                'without it, so play_zoom_tanks has to run first.')

        others = VGroup(*[circle for side in range(2)
                          for slot, circle in enumerate(self.player_circles[side])
                          if not (side == CONTRAST_TEAM
                                  and slot in [self._slot_of(CONTRAST_TEAM, name)
                                               for name in CONTRAST_PLAYERS])])
        self.play(others.animate.set_opacity(DIMMED_OPACITY), run_time=0.6)

        league_height = TANK_BASELINE_Y + self._height_of_percent(
            self.shooting['league_percent'])
        centre_x = self._tank_centre_x(CONTRAST_TEAM)
        self.pull_marks = VGroup()
        # Kept so the closing beat can point back at them when the volume factor is named.
        self.contrast_circles = VGroup(*[
            self.player_circles[CONTRAST_TEAM][self._slot_of(CONTRAST_TEAM, name)]
            for name in CONTRAST_PLAYERS
        ])

        for order, name in enumerate(CONTRAST_PLAYERS):
            slot = self._slot_of(CONTRAST_TEAM, name)
            circle = self.player_circles[CONTRAST_TEAM][slot]
            share, deviation, pull = self._pull_of(CONTRAST_TEAM, name)

            # What the circle already shows, said in numbers: how big it is, and how far its
            # level sits from everyone else's.
            caption = VGroup(
                Text(f'{share:.1%} of the shots', font_size=14, color=GREY_B),
                Text(f'{deviation:+.3f}'.replace('0.', '.'), font_size=18, color=WHITE),
            ).arrange(DOWN, buff=0.06).next_to(circle, UP, buff=0.14)

            # And the product, as a bar standing on the league line inside its own team's tank.
            bar_height = abs(pull) / (2 * ZOOM_HALF_BAND) * TANK_HEIGHT
            bar = Rectangle(
                width=PULL_BAR_WIDTH, height=bar_height,
                stroke_color=WHITE, stroke_width=2,
                fill_color=WHITE, fill_opacity=0.35,
            ).move_to([centre_x + (order - 0.5) * (PULL_BAR_WIDTH + 0.22),
                       league_height + (1 if pull > 0 else -1) * bar_height / 2, 0])

            self.play(FadeIn(caption), run_time=0.55)
            self.play(Create(bar), run_time=0.75)
            self.pull_marks.add(caption, bar)
            self.wait(0.35)

        self.wait(0.9)

    def play_volume_formula(self) -> None:
        """The Z-score with the volume factor in front of it, which is the whole conclusion.

        Built as three expressions side by side rather than one. Each is valid LaTeX on its own,
        so the factor can be pointed at later by name -- where one expression would have to be
        picked apart by glyph position, which breaks the moment the formula is edited.
        """
        self.volume_factor = MathTex(r'\frac{v}{\bar{v}}', font_size=54, color=WHITE)
        times = MathTex(r'\cdot', font_size=54, color=GREY_B)
        z_score = MathTex(r'\frac{x - \mu}{\sigma}', font_size=54, color=WHITE)

        self.volume_formula = VGroup(self.volume_factor, times, z_score)
        self.volume_formula.arrange(RIGHT, buff=0.24).move_to(VOLUME_FORMULA_CENTRE)

        self.play(Write(self.volume_formula), run_time=1.5)
        self.wait(0.8)

    def play_volume_factor_emphasis(self) -> None:
        """Point at the volume factor, and at the circles whose sizes are what it measures.

        The factor is a player's volume over the average volume, and the circles have been
        sitting there sized by exactly that all along -- so the beat joins the symbol to the
        picture rather than introducing anything new.
        """
        outline = SurroundingRectangle(self.volume_factor, color=WHITE, stroke_width=2,
                                       buff=0.10)
        self.play(Create(outline), run_time=0.7)
        self.play(*[circle.animate.scale(CONTRAST_PULSE)
                    for circle in self.contrast_circles], run_time=0.5)
        self.play(*[circle.animate.scale(1 / CONTRAST_PULSE)
                    for circle in self.contrast_circles], run_time=0.5)
        self.volume_factor_outline = outline
        self.wait(1.0)

    def play_shooting_frame(self) -> None:
        """Put the apparatus up: the vessels, then what is in them, then the players."""
        self.build_shooting_frame()
        self.play(Create(VGroup(*[tank[0] for tank in self.tanks])), run_time=1.0)
        self.play(FadeIn(VGroup(*[tank[1] for tank in self.tanks])), run_time=0.8)
        self.play(FadeIn(self.tank_labels), FadeIn(self.attempts_labels), run_time=0.5)
        self.play(Create(self.league_line), run_time=0.7)
        self.play(FadeIn(self.player_circles, lag_ratio=0.04), run_time=1.4)
        self.wait(0.6)
