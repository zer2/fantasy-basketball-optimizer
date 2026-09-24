"""Most Categories: every way the week can go, and the shortcut that makes it computable.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The argument in four moves:

    One, a real week: nine categories, a winner marked against each, and a verdict. Then the
    same two teams in a different week, going the other way. A landslide last, worth exactly
    what the narrow majority was worth -- which is the payoff the whole format turns on.

    Two, every combination as a row of a table, scrolled. What is being computed is a sum over
    rows, so rows are what to show; the length of the scroll is the cost being objected to.

    Three, the same outcomes redrawn as a walk: a category won steps up, one lost steps down,
    and the majority is simply finishing above the line. Two paths reaching the same height are
    worth the same from there on, so the paths are thrown away and one column of heights is
    carried forward instead -- each category translating that column into the next in a single
    pass. That is the dynamic programme the docs describe in prose.

    Four, the same column answers the question worth asking: how often is THIS category the one
    that decides the matchup? It decides exactly when the other eight leave the walk level, and
    that probability is the gradient the algorithm steps on -- and the reason Most Categories
    punts hardest, since a category already certain either way can never tip anything.

Nothing here is measured against the real objective yet. The win probabilities are stand-ins
chosen to make the shape legible; a prep script comes once the beats are settled.

    manim -ql visualizations/wireframes/most_categories/most_categories.py MostCategories
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (
    Group, VGroup, Line, DashedLine, Rectangle, Prism, Text, MathTex, Dot, Dot3D,
    ImageMobject,
    FadeIn, FadeOut, Create, Write, Transform, LaggedStart, linear,
    DOWN, UP, LEFT, RIGHT,
    BLUE_D, BLUE_B, RED_D, RED_B, GREEN_C, GREY_B, GREY_D, YELLOW, WHITE,
)
from manim import ThreeDScene, DEGREES, OUT                # noqa: E402
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.narration_voice import NarrationVoice         # noqa: E402
from shared.narration_timing import (                     # noqa: E402
    wait_until_phrase, seconds_remaining_until_phrase, snap_to_frames,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── The board ────────────────────────────────────────────────────────────────────────

CATEGORIES = ('Field Goal %', 'Free Throw %', 'Threes', 'Points', 'Rebounds',
              'Assists', 'Steals', 'Blocks', 'Turnovers')
# Stand-in win probabilities, not measured: two the team is nearly certain of, two nearly lost,
# and five genuinely in the balance. The shape is what the last beat needs -- the certain ones
# have to be visibly unable to tip anything.
# A team that has committed: five categories close to locked, four given up. Chosen because the
# last act needs the nine tipping points to DIFFER. On a perfectly balanced team they are
# identical by symmetry (27.34% each), and on a mildly tilted one they span only 1.15x -- nine
# dots at the same height, which would say every category is equally decisive. This build spans
# 1.87x, and the categories it kept are the decisive ones, which is the point.
# Nothing above 0.70, because nothing in a real week goes above it. A category is one team's
# nine players against another's over a few games, and even a lopsided matchup lands nearer two
# to one than nineteen to one -- the old profile ran 0.95 down to 0.04, which is not a punt, it
# is a certainty wearing a punt's shape.
#
# It distorted everything downstream. The enumeration's top row took 38.7% of the whole sum, so
# a table built to show a sum showed one row and a rounding error; the cut's convolution had one
# pairing carrying 96% of its weight; and the walk never really wandered. At these rates the top
# row is 1.8% and the sum is a sum.
WIN_CHANCES = (0.70, 0.68, 0.66, 0.64, 0.62, 0.44, 0.40, 0.36, 0.32)
MAJORITY = len(CATEGORIES) // 2 + 1

TEAM_LABELS = ('Team 1', 'Team 2')

# The opening scoreboard: categories down the middle, a team either side.
BOARD_TOP_Y = 2.35
BOARD_ROW_GAP = 0.46
BOARD_HALF_WIDTH = 2.6
BOARD_BOTTOM_Y = -2.6

# Three weeks between the same two teams: a majority, the other way, and a landslide worth
# exactly the same as the majority.
# Weeks between the same two teams, cycled at the top of the scene. One is not enough to show
# what there is a probability OF -- the same teams produce a different result every week, and
# the majority falls either way, sometimes by one category and sometimes by three.
# Opening on the five-four. It is the result the whole scene is about -- the narrowest majority,
# which pays exactly what a sweep pays -- so it is what the words 'a majority of categories'
# should be putting on screen, not a comfortable six-three that makes the margin look incidental.
ROTATING_WEEKS = (
    (1, 0, 1, 1, 0, 1, 0, 1, 0),   # five of nine, the narrowest majority there is
    (1, 1, 0, 1, 1, 0, 1, 0, 1),   # six of nine
    (0, 1, 0, 0, 1, 1, 0, 0, 1),   # four of nine
    (0, 0, 1, 0, 1, 0, 0, 1, 0),   # three of nine
    (1, 1, 1, 0, 1, 1, 0, 1, 0),   # six of nine again, arrived at differently
)
# The pair that makes the payoff argument: a comfortable win and the narrowest possible
# majority, worth exactly the same. Ending on the narrow one is the point -- it is the outcome
# the algorithm actually plays for, and the wide one is only there to be matched by it.
#
# Seven-two rather than nine-nothing. A clean sweep of every category is an outlier nobody's
# week looks like, and an argument about what a win is worth lands better against a result a
# viewer recognises than against one they never see.
COMFORTABLE_WEEK = (1, 1, 1, 0, 1, 1, 0, 1, 1)   # seven of nine
NARROW_WEEK      = (1, 0, 1, 1, 0, 1, 0, 1, 0)   # five of nine, and worth the same

# How long ONE week owns the board, measured start to start, and how much of that is spent
# changing over. Every example gets the same span: the first is dealt out category by category
# and the rest arrive at once, but they are on screen equally long, so no result reads as more
# important than another because it happened to linger.
#
# The board used to clear, pause, and redraw, which left it empty for most of every cycle --
# the rotation is meant to show results changing, and what it showed was an empty scoreboard
# flashing between them.
# The least an example can be worth having on screen at all, used only to decide HOW MANY of
# them fit in the line. What each one actually gets is divided out from the time available.
# How far ahead of the words a number that the voice is about to read aloud should be up.
# Comfortably more than the fade that puts it there, so it has settled rather than just landed.
NUMBER_LEAD_SECONDS = 1.2
# The walk draws itself across the clause that describes taking a step, starting a little before
# the words so the first categories are already going down as they are spoken.
WALK_DRAW_LEAD_SECONDS = 0.8
WALK_DRAW_SECONDS = 4.0
# The two steps the line names one at a time, before the lattice they generate.
FIRST_STEP_SECONDS = 0.55
# Each column starts drawing when the one before it is this far along. Below one, so the front
# of the lattice reads as a wave moving right rather than as nine separate drops.
WALK_DRAW_OVERLAP = 0.7
# What the lattice drops to once the spheres take the foreground: still legible as the ground
# the distribution is measured over, well back from the dots themselves.
WALK_PATH_OPACITY = 0.45
# What the ramp drops to once the spheres take the foreground. Higher than the old line
# lattice needed: a solid set back too far stops reading as a surface at all.
WALK_DIMMED_RAMP_OPACITY = 0.55
# The walks are SOLID. Every step is a beam with width across its own plane and depth standing
# out of it, so the two sweeps sit in space as objects rather than as two drawings at different
# depths -- which is the only way a tilt reads as a tilt.
BEAM_THICKNESS = 0.045
BEAM_DEPTH = 0.42
FORWARD_RAMP_COLOUR = '#2a3444'
BACKWARD_RAMP_COLOUR = '#2c3c2e'
# The level line lies ON the ramp face. Given depth it became a pipe hovering over the walk,
# and given one length it was occluded by every beam nearer the camera than its midpoint and
# vanished part way along -- so it is flat, and segmented one piece per step.
RAIL_COLOUR = '#7e8691'
RAIL_THICKNESS = 0.075
# Capped below the lattice spacing. At 0.05 + 0.18*sqrt a certain outcome came out 0.46 across
# against a 0.34 gap between rows, so the heavy balls overlapped their neighbours and the sweep
# read as lumpy rather than as a distribution.
SPHERE_MIN_RADIUS = 0.042
SPHERE_MAX_RADIUS = 0.145
# The cut is a plane passing through BOTH ramps, which is what the line describes.
CUT_PLANE_COLOUR = '#c8b24a'
# Where the camera looks during the tilted act. The pair of ramps hangs BELOW the forward plane
# -- the backward one is set back in depth and depth projects downward -- so an untouched frame
# centre left the whole arrangement sitting eighty pixels low. Once the suffix is glued forward
# everything is in the forward plane again and rides high, so the centre moves back.
TILTED_FRAME_CENTRE = [0.0, -1.45, 0.0]
GLUED_FRAME_CENTRE = [0.0, -0.55, 0.0]
# The nine categories listed down the left, each in the colour of the walk that accounts for it.
CUT_LEGEND_X = -5.75
CUT_LEGEND_GAP = 0.32
# How long a week sits on the board, set outright rather than divided out of whatever a line
# has left over. EVERY example in the opening gets this, across BOTH of its lines: the rotation,
# the wide win and the narrow win alike. Dividing each line's own time among its own examples
# gave two different rhythms -- the rotation at 2.4 seconds and the pair after it at 8.5 -- and
# the opening reads as one sequence, so it needs one rhythm.
EXAMPLE_HOLD_SECONDS = 2.0
ROTATION_SWAP_SECONDS = 0.5
# The first week is dealt one category at a time, verdict last, which the other four do not
# need. Measured off an intro-only render: with the verdict as a separate fade afterwards, the
# first week sat on screen eight tenths of a second longer than any other.
FIRST_DEAL_SECONDS = 1.9
# The deal is not finished when the board is: the ninth mark lands, and the verdict is still
# arriving behind it. That tail is time the first week is fully readable, so it counts against
# its hold -- without this it measured 2.73s against 2.33s for every week after it.
FIRST_DEAL_TAIL_SECONDS = 0.4
# How long the board takes to clear once its line has finished.
BOARD_CLEAR_SECONDS = 0.7
# The least a repeat cut can be given and still read as one. The slide's line is short, so each
# of its cuts gets whatever it leaves once the readouts are paid for.
RECUT_MINIMUM_SECONDS = 1.4
# Flattening the camera and clearing the frame, which the slide's line pays for out of its own
# tail so the punting line opens on a settled picture.
SCENE_CHANGE_SECONDS = 1.9

# The table of outcomes. Only the first rows are built -- enough to scroll convincingly without
# drawing all 512, which would cost minutes of render for no extra argument.
# The enumeration is the full 2^8 = 256 ways to take the majority -- winning scenarios only,
# ordered by probability, and their products really do sum to the 63.9% the walk arrives at
# independently. What gets BUILT is the part the scroll can reach.
#
# The scroll travels TABLE_SCROLL_ROWS_PER_SECOND rows a second for as long as its line lasts,
# and it can only travel over rows that were baked. So the count below has to cover
#
#     rows the scroll reaches  +  TABLE_VISIBLE_ROWS still on screen at the end
#
# and everything past that is off the bottom of the frame for the whole act, never seen, and
# still drawn by Manim on every single frame. Building all 256 costs about seven and a half
# minutes of render for the scroll alone against three for a hundred, for an identical picture.
#
# Raised from 72 after dropping the 2^9 counter: that counter used to take 2.2 seconds off the
# end of the line, and handing those seconds back to the scroll bought 4 more rows of travel,
# which put it 1 row over what 72 could supply. play_the_table refuses to run rather than
# scrolling off the end of the image, so this is a loud failure -- but only a full render
# reaches it, since every act short of one starts after the table.
TABLE_SCENARIOS = 2 ** (len(CATEGORIES) - 1)
TABLE_ROWS_BUILT = 84
# How long one chunk of the scroll runs. The scroll is one long translation, but it is played
# in chunks so the set of rows in the scene can be refreshed between them -- short enough that
# few off-screen rows are carried, long enough not to drown the render in partial files.
# Where the baked enumeration lives, and the record of what it was baked from.
TABLE_IMAGE_PATH = Path(__file__).resolve().parent / 'outcome_table.png'
TABLE_RECORD_PATH = Path(__file__).resolve().parent / 'outcome_table.json'
# The image carries a hair of margin around the rows, so its top edge sits that much above where
# the first row's top used to be.
MARGIN_ABOVE_TABLE = 0.06
TABLE_VISIBLE_ROWS = 10
# Slower than it was. Each cell now carries a number, and a rate that was fine for reading
# colours goes past too quickly to read digits.
TABLE_SCROLL_ROWS_PER_SECOND = 3.0
# Wide enough to write the probability inside. The cells were squares too small to hold a
# number, which left the rows saying only WHICH outcome, never how likely it was -- and the
# line is 'each has its own probability'.
TABLE_CELL_WIDTH = 0.46
TABLE_CELL_HEIGHT = 0.34
TABLE_CELL_FONT = 13
TABLE_ROW_GAP = 0.44
# How long the enumeration takes to leave, once its line has finished scrolling.
TABLE_CLEAR_SECONDS = 0.8
TABLE_TOP_Y = 3.4

# The walk the tree collapses into: one column per category, net position up for a category won
# and down for one lost. Nine steps is odd, so the walk can never finish level -- it is above the
# line exactly when five or more were won, which is what makes "majority" a side of the picture.
# Kept near square. At ten units wide against five tall the walk sprawled sideways and the
# steps read as shallow; the shape of a random walk is easier to see when a step up is about as
# big as a step along.
WALK_LEFT_X = -3.7
WALK_RIGHT_X = 3.7
WALK_CENTRE_Y = -0.30
WALK_UNIT_Y = 0.34
# Where the cut opens first, and where it slides to -- one kept category, then two abandoned.
CUT_CATEGORY = 2
SLIDE_TO = (5, 0)

# The closing example: five categories held, four given up. Which one is taken away decides
# whether the remaining eight are level, and that is the whole punting argument.
# The three cases the toggle panel steps through: the other eight level, then already won,
# then already lost. Only the first leaves the ninth category mattering at all.
# The two sweeps sit on parallel planes, so tilting the camera separates them in depth rather
# than leaving the backward one nudged sideways and overlapping the forward one. Flat-on
# (phi = 0) the scene renders exactly as a 2D one, which is what every other act wants.
WALK_BACKWARD_Z = -2.6
# Manim's 3D camera sorts every mobject SEPARATELY by its distance from the camera, and at this
# angle height counts for more than depth: one unit of y is worth 0.765 of depth against 0.616
# for a unit of z, and the walk fans 6.1 units tall while the two planes are only 2.6 apart. So
# a low beam on the BACK ramp sorted nearer the camera than a high beam on the front one, and
# wherever the two overlapped on screen the back walk painted over the front -- inconsistently,
# because which won depended on the heights that happened to meet.
#
# The two walks are flat parallel planes that never interpenetrate, so which one is in front is
# a CONSTANT. These anchors state it. Manim reads a mobject's depth from its z_index_group
# rather than from itself, so handing every piece of a walk the same anchor makes the whole walk
# take one depth and sort as one object instead of as ninety competing ones. Never added to the
# scene -- they exist only to be asked for their centre.
FORWARD_DEPTH_ANCHOR = Dot(point=[0.0, 0.0, 0.0])
BACKWARD_DEPTH_ANCHOR = Dot(point=[0.0, 0.0, WALK_BACKWARD_Z])
# The cut is glass laid over both walks, so it sits nearer than either.
CUT_DEPTH_ANCHOR = Dot(point=[0.0, 0.0, 1.0])
# What draws over what WITHIN one walk, once the anchor has made the walk sort as a unit: the
# rail is painted on the ramp face, and the spheres stand on top of both.
RAIL_LAYER, RAMP_LAYER, SPHERE_LAYER = 0, 1, 2
# How finely each probability sphere is built. Low on purpose: there are around a hundred of
# them per sweep and the scene renders them for the whole act.
# Six, not eight. A Dot3D is a Surface of resolution-squared quad faces, so this is the
# single biggest lever on render cost: at 8 the two walks put 8,578 mobjects on screen and
# 7,040 of them are sphere faces; at 6 that falls to 5,498. Measured on the walk act, 94s
# against 77s, and a frame rendered at each is indistinguishable magnified six times.
SPHERE_RESOLUTION = 6
# What a column drops to when the cut is not keeping it.
DIMMED_OPACITY = 0.12
# The majority total goes in the bottom-left corner. The walk fans out to the RIGHT from a
# single point at the left, so that corner is the one part of the frame it never occupies.
MAJORITY_READOUT_POSITION = [-3.5, -3.55, 0.0]
# How wide the shaded majority band is: one column's worth, centred on the last step.
WALK_COLUMN_WIDTH = 0.62
# Enough tilt to read as a tilt. At thirty-four degrees and eighty-four the two planes were
# very nearly edge-on to the camera and the whole thing came back looking like flat artwork --
# the depth was there in the geometry and invisible in the picture. At sixty-plus the planes
# foreshorten into slivers and text on them shears badly, so this sits between: the sweeps
# separate clearly and the walk still reads as a walk.
WALK_VIEW_PHI, WALK_VIEW_THETA = 52, -76
# Readouts are pinned in SCREEN space, not on the planes: at any camera angle a line of text
# lying on a tilted plane shears, and pinning only its orientation left the glyphs spread apart.
READOUT_SCREEN_Y = -3.05
CUT_LABEL_SCREEN_Y = 3.25
FLAT_VIEW_PHI, FLAT_VIEW_THETA = 0, -90

# The focus sits in the middle of the row, so the other eight fall four either side of it.
FOCUS_INDEX = 4
PANEL_CELL_WIDTH = 0.82
PANEL_CELL_HEIGHT = 0.52
PANEL_CELL_GAP = 0.95
PANEL_ROW_Y = 1.35
# The panel's two readouts, side by side. The line talks about the chance of the majority AND
# about the gradient, and they are different numbers in every case the beat shows: level gives
# a chance of p against a gradient of 1, and a decided board gives a chance of 1 or 0 against a
# gradient of 0 either way. One readout could only ever be right about one of them.
CHANCE_READOUT_X = -2.75
GRADIENT_READOUT_X = 2.75
READOUT_ROW_Y = PANEL_ROW_Y - 1.75
# The identity the beat is built to land, assembled a term at a time as the line states them.
EQUATION_Y = -2.2
EQUATION_FONT = 40
# The balance sits above the row, in the space the old heading used to take. Wins on the right,
# losses on the left, and level exactly when the other eight split four-four -- which is the
# whole reason the quantity is called a TIPPING point, and a word worth earning before it is
# used. The tilt is capped so a six-two board leans hard without the beam standing on end.
BALANCE_Y = 2.55
BALANCE_WIDTH = 3.1
BALANCE_MAX_TILT = 26

LEVEL_CASE   = (1, 1, 1, 1, 0, 0, 0, 0)
# Three more ways the other eight can come out level. The point of the closing beat is that
# these are all DIFFERENT scenarios giving the SAME answer, so it needs more than one of them.
OTHER_LEVEL_CASES = (
    (0, 1, 0, 1, 1, 0, 1, 0),
    (1, 0, 0, 1, 0, 1, 1, 0),
    (0, 0, 1, 1, 1, 0, 0, 1),
)
if any(sum(case) != MAJORITY - 1 for case in (LEVEL_CASE,) + OTHER_LEVEL_CASES):
    raise ValueError('every level case has to split the other eight four-four: the closing '
                     'beat puts them up one after another and claims each one gives p')
DECIDED_WON  = (1, 1, 1, 1, 1, 1, 0, 0)
DECIDED_LOST = (1, 1, 0, 0, 0, 0, 0, 0)

PUNT_BAR_HEIGHT = 1.9
# The board is an EXTREME team, and deliberately not the one the walk used, because the effect
# it claims does not survive realistic odds: at the walk's own rates, capped at 0.70, every
# category's tipping point lands between 26.6% and 28.6% and there is nothing to show.
#
# Four categories won decisively, four lost decisively, one a coin flip. This says the thing
# better than the old five-four board did, because every DECISIVE category comes out at exactly
# the same tipping point whether it was won or lost -- 43.5% against the average category's
# 69.3% -- so the picture says the asymmetry is about being AVERAGE rather than about winning.
PUNT_BOARD_ODDS = (0.95,) * 4 + (0.05,) * 4 + (0.50,)
# Which category is taken off the board in each case: the average one, then a decisive one.
AVERAGE_CATEGORY = 8
EXCLUDED_DECISIVE = 2
# How tall the tallest tipping-point bar is drawn, and where the strip sits under the board.
TIPPING_STRIP_HEIGHT = 0.85
# Dropped from -1.55 to make room for the strip's caption to sit ABOVE its bars, the way
# the board's does. Both figures then read the same way down the frame -- title, bars,
# numbers -- instead of the top one being titled from above and the bottom one from below.
TIPPING_STRIP_BASE = -2.05
TIPPING_STRIP_COLOUR = '#c8873a'
# How many of the 256 scenarios are drawn before the table is summarised rather than continued.


def anchor_walk_depth(pieces, anchor, layer: int):
    """Give every piece of a walk one shared depth, and a fixed order among its own parts.

    The anchor decides which WALK is in front; the layer decides what is on top inside it.
    Manim sorts by z_index first and then stably by depth, so pieces sharing an anchor keep
    their layer order and pieces on different anchors are separated by the anchor alone.
    """
    for piece in pieces.get_family():
        piece.z_index_group = anchor
        piece.z_index = layer
    return pieces


def build_outcome_table() -> VGroup:
    """One row per WINNING outcome: nine cells, their nine chances, and their product.

    Only the winning scenarios, because those are the only ones being added up -- the line
    says so itself ("two to the eighth if we only consider the winning scenarios"). A losing
    row contributes nothing to the sum, so showing it invites the viewer to add it in.

    Ordered by probability rather than by binary counting. Counting down from all-wins puts
    the least likely scenarios first: this team wins its five strong categories and loses
    its four weak ones, so a clean sweep is a one-in-seventy-thousand row, and the first
    screenful would have read 0.001%, 0.033%, 0.022% -- true, and useless as an argument
    that these add up to anything. Sorted, the first row IS the most likely way to take the
    majority, and the column visibly decays from there.

    The products are the whole point of the act: each row's nine chances multiplied, which
    is the number the next act finds a cheaper way to total. All 256 sum to 63.9%, the
    figure the walk arrives at independently.
    """
    scenarios = []
    for number in range(2 ** len(CATEGORIES)):
        bits = [(number >> shift) & 1 for shift in range(len(CATEGORIES) - 1, -1, -1)]
        if sum(bits) < MAJORITY:
            continue
        product = 1.0
        for bit, chance in zip(bits, WIN_CHANCES):
            product *= chance if bit else 1.0 - chance
        scenarios.append((product, bits))
    scenarios.sort(key=lambda entry: entry[0], reverse=True)

    rows = VGroup()
    for product, bits in scenarios[:TABLE_ROWS_BUILT]:
        cells = VGroup()
        for bit, chance in zip(bits, WIN_CHANCES):
            cell = Rectangle(width=TABLE_CELL_WIDTH, height=TABLE_CELL_HEIGHT,
                             fill_color=BLUE_D if bit else GREY_D,
                             fill_opacity=1.0, stroke_width=0)
            # The probability of THIS cell's outcome: a blue cell carries the chance of
            # winning that category, a grey one the chance of losing it. They are
            # complements, so a column alternates between .12 and .88 -- which looks odd
            # until you see that it is what makes the row MULTIPLY to the product beside
            # it. Showing the win chance in every cell would line the columns up and quietly
            # break the arithmetic the whole act is about.
            landed = chance if bit else 1.0 - chance
            cells.add(VGroup(cell, Text(f'{landed:.2f}'.lstrip('0'),
                                       font_size=TABLE_CELL_FONT,
                                       color=WHITE if bit else GREY_B)
                            .move_to(cell.get_center())))
        cells.arrange(RIGHT, buff=0.08)
        # The product, which is what the line asks to be added up. The verdict column it
        # replaces said 'win' on every row here, every row being a win.
        total = Text(f'{product:.3%}', font_size=19, color=BLUE_B)
        rows.add(VGroup(cells, total).arrange(RIGHT, buff=0.42))
    # Left-aligned, so the nine cells sit in nine straight columns. Stacked centred, a row
    # whose product reads 38.739% is wider than one reading 0.932%, and centring shifted
    # every row's cells sideways by half the difference -- the grid was not a grid.
    rows.arrange(DOWN, buff=TABLE_ROW_GAP - TABLE_CELL_HEIGHT, aligned_edge=LEFT)
    rows.move_to([0.0, TABLE_TOP_Y, 0.0], aligned_edge=UP)
    return rows

# ── Act three: the tree is a walk, and the walk is one column ─────────────────────


class MostCategories(VoiceoverScene, ThreeDScene):
    """The 512-leaf tree, its collapse into a tally, and the tipping point that falls out."""

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.play_a_week()
        self.play_the_table()
        self.play_walk()
        self.play_the_slope()
        self.play_what_tipping_means()
        self.play_tipping_point()

    # -- Act one: a week, and who took it ----------------------------------------------

    def play_a_week(self) -> None:
        """One matchup, category by category, so the thing being counted is concrete first.

        The scene used to open on an empty frame while the first line was spoken. It opens on a
        scoreboard instead: nine categories, a winner marked against each, and a verdict. Two
        different weeks between the same teams make the point that the same teams do not produce
        the same result -- which is what there is a probability OF.
        """
        with self.voiceover(text=NARRATION['a_week']) as tracker:
            self.board = self.build_scoreboard()
            self.play(FadeIn(self.board), run_time=1.0)

            # On the words that name it. Held back to 'five out of nine' the board sat blank
            # through the clause the scene is actually about.
            wait_until_phrase(self, tracker, 'majority of categories')
            # The first week is dealt out category by category; the rest cycle, because by then
            # the viewer knows how to read the board and what is worth showing is that the
            # answer keeps changing.
            # LaggedStart, not a lag_ratio handed to play(): play() sets that on each FadeIn
            # separately, and on a single dot it does nothing -- so this deal, described here
            # as one category at a time since the scene was written, had all nine marks fading
            # in together.
            self.verdict = self.build_verdict(ROTATING_WEEKS[0])
            self.play(LaggedStart(*[FadeIn(mark) for mark in self.mark_week(ROTATING_WEEKS[0])],
                                  FadeIn(self.verdict), lag_ratio=0.12),
                      run_time=FIRST_DEAL_SECONDS)
            # The deal is not over when the board is: the ninth mark lands and the verdict is
            # still arriving behind it, and that tail is already dwell.
            self.wait(EXAMPLE_HOLD_SECONDS - FIRST_DEAL_TAIL_SECONDS)

            self.rotation_index = 1
            self.rotate_weeks(tracker, reserve=0.0)

        with self.voiceover(text=NARRATION['all_or_nothing']) as tracker:
            # A wide win and the narrowest one, at the same dwell as every other week. That
            # they are worth the same is the entire claim, so neither can sit longer than the
            # other -- or, just as badly, longer than the rotation they follow.
            self.play(*self.swap_week(COMFORTABLE_WEEK), run_time=ROTATION_SWAP_SECONDS)
            self.wait(EXAMPLE_HOLD_SECONDS)
            self.play(*self.swap_week(NARROW_WEEK), run_time=ROTATION_SWAP_SECONDS)
            self.wait(EXAMPLE_HOLD_SECONDS)

            # The line runs on past the pair, so the rotation continues at the same rhythm
            # rather than parking one week on the board for the rest of it.
            self.rotate_weeks(tracker, reserve=BOARD_CLEAR_SECONDS)
            self.play(FadeOut(self.board), FadeOut(self.week_marks),
                      FadeOut(self.verdict), run_time=BOARD_CLEAR_SECONDS)

    def build_scoreboard(self) -> VGroup:
        """Two team headings and the nine categories between them."""
        rows = VGroup()
        for index, name in enumerate(CATEGORIES):
            y = BOARD_TOP_Y - index * BOARD_ROW_GAP
            rows.add(Text(name, font_size=22, color=GREY_B).move_to([0.0, y, 0.0]))
        headings = VGroup(
            Text(TEAM_LABELS[0], font_size=26, color=BLUE_B)
                .move_to([-BOARD_HALF_WIDTH, BOARD_TOP_Y + 0.65, 0.0]),
            Text(TEAM_LABELS[1], font_size=26, color=RED_B)
                .move_to([BOARD_HALF_WIDTH, BOARD_TOP_Y + 0.65, 0.0]),
        )
        return VGroup(rows, headings)

    def mark_week(self, won_by_first) -> VGroup:
        """A marker against each category on the side of whoever took it."""
        self.week_marks = VGroup()
        for index, first_won in enumerate(won_by_first):
            y = BOARD_TOP_Y - index * BOARD_ROW_GAP
            x = -BOARD_HALF_WIDTH if first_won else BOARD_HALF_WIDTH
            self.week_marks.add(Dot(point=[x, y, 0.0], radius=0.15,
                                    color=BLUE_D if first_won else RED_D))
        return self.week_marks

    def rotate_weeks(self, tracker, reserve: float) -> None:
        """Keep putting weeks up at the scene's one rhythm until the line runs out.

        Every week costs the same: one swap, one hold. Whatever is left when another will not
        fit lands on the last week shown, and that is at most a single cycle -- against the six
        and a half seconds the old break-out-and-wait used to leave it holding.
        """
        cycle = ROTATION_SWAP_SECONDS + EXAMPLE_HOLD_SECONDS
        # Rounded to the NEAREST whole week rather than stopping at the last one that fits.
        # Stopping short left up to a full cycle unspent, and a voiceover block waits for its
        # audio, so all of it landed on the last week -- measured at five seconds against two
        # and a half for every other. Rounding lets the line overrun by at most half a cycle
        # instead, which the next line simply starts after.
        fits = max(0, round((tracker.get_remaining_duration() - reserve) / cycle))
        for _ in range(fits):
            week = ROTATING_WEEKS[self.rotation_index % len(ROTATING_WEEKS)]
            self.rotation_index += 1
            self.play(*self.swap_week(week), run_time=ROTATION_SWAP_SECONDS)
            self.wait(EXAMPLE_HOLD_SECONDS)

    def swap_week(self, week) -> list:
        """Put a different week on the board without the board going empty in between.

        The old marks leave as the new ones arrive, rather than the board being cleared and
        then redrawn. Cleared first, it spent most of every cycle showing nothing, and a
        rotation whose point is that the result keeps changing was mostly darkness.
        """
        outgoing_marks, outgoing_verdict = self.week_marks, self.verdict
        incoming_marks = self.mark_week(week)
        self.verdict = self.build_verdict(week)
        return [FadeOut(outgoing_marks), FadeOut(outgoing_verdict),
                FadeIn(incoming_marks), FadeIn(self.verdict)]

    def build_verdict(self, won_by_first) -> VGroup:
        """The tally and who it gives the matchup to."""
        taken = sum(won_by_first)
        first_wins = taken > len(CATEGORIES) / 2
        winner = TEAM_LABELS[0] if first_wins else TEAM_LABELS[1]
        colour = BLUE_B if first_wins else RED_B
        return VGroup(
            Text(f'{taken} - {len(CATEGORIES) - taken}', font_size=40, color=colour),
            Text(f'{winner} wins  ->  1 win', font_size=26, color=colour),
        ).arrange(DOWN, buff=0.22).move_to([0.0, BOARD_BOTTOM_Y, 0.0])

    # -- Act two: everything that could happen, as rows --------------------------------

    def play_the_table(self) -> None:
        """The 512 combinations as a table, scrolled, so the sum being asked for is visible.

        This replaces a binary tree. The tree showed the same 512 outcomes but spent its length
        on the branching, which is not the thing being computed -- what is being computed is a
        sum over rows, so rows are what to show.
        """
        with self.voiceover(text=NARRATION['the_table']) as tracker:
            # One baked image rather than eight hundred vector objects. The table is a still
            # picture that only translates, and Manim redraws every mobject on every frame in
            # shot or not, so scrolling it as objects cost 170 seconds of a 258 second render
            # -- in this single animation. See prepare_outcome_table.py.
            self.table = self.build_table_image()
            self.play(FadeIn(self.table), run_time=1.2)
            wait_until_phrase(self, tracker, 'each of these scenarios')
            self.wait(1.0)

        with self.voiceover(text=NARRATION['how_many_rows']) as tracker:
            # Scrolled rather than paged: the length of it is the argument, and a scroll is the
            # only way to show length without drawing 512 rows.
            # Scrolled at a fixed RATE for whatever time the line leaves, rather than a fixed
            # distance in a fixed time. The enumeration is longer than the sentence either way,
            # so the only question is when it stops, and it should stop when the line does.
            scroll_seconds = max(2.0, tracker.get_remaining_duration() - TABLE_CLEAR_SECONDS)
            travel = TABLE_ROW_GAP * TABLE_SCROLL_ROWS_PER_SECOND * scroll_seconds
            # Measured off the image, not off a row count: the table is one picture now, and
            # len() of it counts submobjects rather than rows -- which made this guard compute
            # minus nine rows of travel and refuse a scroll that fits comfortably.
            available = self.table.height - TABLE_VISIBLE_ROWS * TABLE_ROW_GAP
            if travel > available:
                raise RuntimeError(
                    f'the scroll needs {travel / TABLE_ROW_GAP:.0f} rows of travel but the '
                    f'baked table only holds {available / TABLE_ROW_GAP:.0f}; raise '
                    f'TABLE_ROWS_BUILT and re-run prepare_outcome_table.py')
            self.play(self.table.animate.shift(UP * travel),
                      run_time=scroll_seconds, rate_func=linear)
            self.play(FadeOut(self.table), run_time=TABLE_CLEAR_SECONDS)

    def build_table_image(self) -> ImageMobject:
        """The enumeration, as the image prepare_outcome_table.py baked from these same numbers.

        Fails loudly rather than scrolling a stale picture: the odds and the row count are
        written into the record beside the image, and an edit to either without re-running the
        prep script would otherwise show a table of the old probabilities under the new ones.
        """
        if not TABLE_IMAGE_PATH.exists():
            raise FileNotFoundError(
                f'{TABLE_IMAGE_PATH.name} is missing. Run '
                f'python visualizations/wireframes/most_categories/prepare_outcome_table.py')
        record = json.loads(TABLE_RECORD_PATH.read_text(encoding='utf-8'))
        baked = (record['rows'], record['row_gap'], tuple(record['win_chances']))
        wanted = (TABLE_ROWS_BUILT, TABLE_ROW_GAP, WIN_CHANCES)
        if baked != wanted:
            raise RuntimeError(
                f'{TABLE_IMAGE_PATH.name} was baked from {baked} but the scene now wants '
                f'{wanted}. Re-run prepare_outcome_table.py.')

        table = ImageMobject(str(TABLE_IMAGE_PATH))
        table.height = record['scene_height']
        table.move_to([0.0, TABLE_TOP_Y + MARGIN_ABOVE_TABLE, 0.0], aligned_edge=UP)
        return table

    def play_walk(self) -> None:
        """Re-draw the same 512 outcomes as a walk, then stop following paths at all.

        A win steps up, a loss steps down, so after nine odd-numbered steps the walk finishes
        above zero exactly when five or more categories were won. The majority stops being a
        counting rule and becomes a side of the picture, which is what lets the next move land:
        two paths that reach the same height are worth the same from there on, so the algorithm
        can throw the paths away and keep one column of heights.
        """
        with self.voiceover(text=NARRATION['dynamic']) as tracker:
            # Up from the first word, with the odds already on it. Held back to 'only matters
            # how many' this beat opened on five seconds of black, and the four orderings only
            # became a comparison once their probabilities were written -- four scenarios at a
            # quarter each is the thing the three numbers on the right replace.
            pairs = self.build_two_category_example()
            # The left half arrives as one thing, because it IS one thing: four orderings,
            # what each of them counts as, and the quarter chance each carries. Bringing the
            # names in afterwards made them read as a second idea when they are a column of the
            # same table.
            #
            # One fade with no stagger. A lag_ratio handed to play() is set on each animation
            # separately and then spreads across that animation's own mobject family -- and
            # these families are different sizes, so flat labels finished arriving while rows
            # of cells were still coming.
            self.play(FadeIn(VGroup(pairs['orders'], pairs['same_count'], pairs['chances'])),
                      run_time=1.2)

            # What the line is actually claiming: the two middle orderings are different ways
            # of landing in the same place, so they can be carried as one number.
            wait_until_phrase(self, tracker, 'only matters how many')
            self.play(FadeIn(pairs['same_brace']), run_time=0.7)

            # The tally is the distribution the line says to keep track of. It appears empty and
            # is filled in later, so the four orderings on the left visibly collapse into three
            # numbers on the right rather than being replaced by them.
            wait_until_phrase(self, tracker, 'keeping track of the distribution')
            self.play(FadeIn(pairs['tally']), run_time=0.9)

            # Each number is fully UP before it is spoken. Anchoring to an earlier phrase and
            # hoping the arithmetic worked out left them landing late twice over; this uses the
            # lead the timing helper was built for, so the trigger stays tied to the words that
            # actually name the number and moves with them if the line is rewritten.
            wait_until_phrase(self, tracker, '50 percent chance of being tied',
                              lead_seconds=NUMBER_LEAD_SECONDS)
            self.play(FadeIn(pairs['even']), run_time=0.45)

            wait_until_phrase(self, tracker, '25 percent chances',
                              lead_seconds=NUMBER_LEAD_SECONDS)
            self.play(FadeIn(pairs['edges']), run_time=0.45)

            # Only the orderings go here, which is exactly what the clause says: there is no
            # need to keep track of WHICH way the first two landed. The three numbers they
            # collapse into stay up, because they are what the sentence says to remember.
            wait_until_phrase(self, tracker, 'no need to keep track')
            self.play(FadeOut(pairs['orders']), FadeOut(pairs['same_count']),
                      FadeOut(pairs['chances']), FadeOut(pairs['same_brace']), run_time=0.8)

        with self.voiceover(text=NARRATION['collapse']) as tracker:
            # The tally holds across the gap between the two lines and clears on the words that
            # introduce the walk. Cleared at the end of its own line instead, the example
            # vanished with five seconds of that line still to run and the walk's bare axis sat
            # alone through them -- and the axis was being drawn through the bottom tally row
            # while it faded, the line sitting at y = -0.30 against that row at y = -0.25.
            self.play(FadeOut(pairs['tally']), FadeOut(pairs['even']),
                      FadeOut(pairs['edges']), run_time=0.7)
            self.lattice = self.build_lattice()
            self.play(FadeIn(self.lattice), run_time=1.0)

            # Drawn one category at a time, from the left, while the line describes taking a
            # step. It used to arrive whole on a single fade, which showed the finished lattice
            # rather than the act of walking it -- and the sentence is entirely about the step.
            # The first two steps are drawn on the words that name them: one beam up on
            # 'upwards', one beam down on 'down'. They are the only two the line describes
            # individually, and everything after is those two repeated -- so everything after
            # arrives as a lattice rather than as steps.
            self.all_paths = self.build_all_paths()
            first_up, first_down = self.all_paths[0][0], self.all_paths[0][1]

            wait_until_phrase(self, tracker, 'one level upwards',
                              lead_seconds=WALK_DRAW_LEAD_SECONDS)
            self.play(Create(first_up), run_time=FIRST_STEP_SECONDS)

            wait_until_phrase(self, tracker, 'one level down',
                              lead_seconds=WALK_DRAW_LEAD_SECONDS)
            self.play(Create(first_down), run_time=FIRST_STEP_SECONDS)

            # LaggedStart so the COLUMNS are what is staggered. A lag_ratio handed to play()
            # is set on each Create separately, which staggered the lines inside every column
            # while all nine columns ran together: the lattice grew from the bottom up
            # everywhere at once rather than from the left.
            self.play(LaggedStart(*[Create(column) for column in self.all_paths[1:]],
                                  lag_ratio=WALK_DRAW_OVERLAP), run_time=WALK_DRAW_SECONDS)

            wait_until_phrase(self, tracker, 'above the middle line')
            self.win_region = self.build_win_region()
            self.play(FadeIn(self.win_region), run_time=0.8)

            # Sized to FINISH on the answer's cue rather than started on a word and left
            # to run as long as it likes. The sweep needs about five seconds and the line only
            # leaves two between 'traversed across' and 'add those probabilities up', so an
            # anchor on the former overran the latter by 2.75s and the answer arrived four
            # seconds after Alistair asked for it. It begins as soon as the winning band is up.
            self.play_column_sweep(
                seconds_remaining_until_phrase(self, tracker, 'add those probabilities up'))

            # The answer, picked out of the last column: the probabilities above the line, added
            # up. The line asks for exactly this and the sweep stopped one step short of it.
            wait_until_phrase(self, tracker, 'add those probabilities up')
            final = self.forward[len(CATEGORIES)]
            gold = self.build_column(final, len(CATEGORIES), majority_colour=YELLOW)
            self.play(Transform(self.forward_columns[-1], gold), run_time=0.8)
            # Down in the bottom-left corner, where the walk has not reached: centred under the
            # middle of the frame it sat among the lower spheres of the middle columns.
            self.majority_readout = Text(
                f'{self.winning_mass(final):.1%} chance of the majority',
                font_size=27, color=YELLOW).move_to(MAJORITY_READOUT_POSITION)
            self.add_fixed_in_frame_mobjects(self.majority_readout)
            self.play(FadeIn(self.majority_readout), run_time=0.9)
            self.wait(0.8)

    def build_two_category_example(self) -> dict:
        """Two coin-flip categories, their four orderings, and the tally they collapse into.

        The whole dynamic programme in miniature: win-then-lose and lose-then-win are different
        orderings and the same result, so only the count is worth carrying forward. The cells
        are the same blue and grey as the scenario table, because they mean the same thing.
        """
        def build_ordering(won, y):
            cells = VGroup(*[
                Rectangle(width=0.5, height=0.36, stroke_width=0, fill_opacity=1.0,
                          fill_color=BLUE_D if bit else GREY_D)
                for bit in won
            ]).arrange(RIGHT, buff=0.1)
            return cells.move_to([-4.2, y, 0.0])

        ORDER_Y = (1.7, 1.0, 0.3, -0.4)
        orders = VGroup(*[build_ordering(won, y) for won, y
                          in zip(((1, 1), (1, 0), (0, 1), (0, 0)), ORDER_Y)])
        same_count = VGroup(*[
            Text(name, font_size=20, color=YELLOW if name == 'one each' else GREY_B)
            .move_to([-2.7, y, 0.0])
            for name, y in zip(('both won', 'one each', 'one each', 'both lost'), ORDER_Y)
        ])
        chances = VGroup(*[
            Text('25%', font_size=20, color=GREY_B).move_to([-1.4, y, 0.0]) for y in ORDER_Y
        ])

        TALLY_Y = (1.35, 0.55, -0.25)
        tally = VGroup(*[
            Text(name, font_size=24, color=YELLOW if name == 'one each' else GREY_B)
            .move_to([2.1, y, 0.0])
            for name, y in zip(('both won', 'one each', 'both lost'), TALLY_Y)
        ])
        even = Text('50%', font_size=34, color=YELLOW).move_to([3.9, TALLY_Y[1], 0.0])
        edges = VGroup(*[
            Text('25%', font_size=28, color=GREY_B).move_to([3.9, y, 0.0])
            for y in (TALLY_Y[0], TALLY_Y[2])
        ])
        # A brace across the two middle rows: the one claim the left half makes on its own,
        # that win-then-lose and lose-then-win are the same result arrived at differently.
        span = VGroup(orders[1], orders[2], same_count[1], same_count[2])
        same_brace = VGroup(
            Line([-4.95, ORDER_Y[1] + 0.22, 0.0], [-4.95, ORDER_Y[2] - 0.22, 0.0],
                 color=YELLOW, stroke_width=2.5),
            Text('same result', font_size=17, color=YELLOW)
            .move_to([-5.05, (ORDER_Y[1] + ORDER_Y[2]) / 2, 0.0]).shift(LEFT * 0.55),
        )
        return {'orders': orders, 'same_count': same_count, 'chances': chances,
                'same_brace': same_brace, 'tally': tally, 'even': even, 'edges': edges}

    # The walk: ten columns (before any category, then after each of the nine) against net
    # position, which runs from -9 to +9 but only ever reaches values of the step's own parity.
    def walk_x(self, step: int) -> float:
        return WALK_LEFT_X + step * (WALK_RIGHT_X - WALK_LEFT_X) / len(CATEGORIES)

    def walk_y(self, net: int) -> float:
        return WALK_CENTRE_Y + net * WALK_UNIT_Y

    def build_lattice(self) -> VGroup:
        """The axis the walk happens on: the level rail, and what each side of it means."""
        zero = self.build_zero_rail(forward=True)
        start = Text('start', font_size=20, color=GREY_B)
        start.next_to([WALK_LEFT_X - 0.3, self.walk_y(0), 0.0], LEFT, buff=0.25)
        return VGroup(zero, start)

    def build_win_region(self) -> VGroup:
        """The top of the LAST column, which is the only place the majority is decided.

        Shaded across the whole walk, this said that being above the line at any point was
        winning -- which is exactly the mistake the scene is built to avoid. A walk that is two
        up after three categories has won nothing; it can still finish below. Only where the
        ninth category lands settles the matchup, so only that column is shaded.
        """
        last_x = self.walk_x(len(CATEGORIES))
        band = Rectangle(
            width=WALK_COLUMN_WIDTH, height=len(CATEGORIES) * WALK_UNIT_Y,
            fill_color=BLUE_D, fill_opacity=0.16, stroke_width=0,
        ).move_to([last_x, self.walk_y(0) + len(CATEGORIES) / 2 * WALK_UNIT_Y, 0.0])
        label = Text('majority', font_size=21, color=BLUE_B)
        label.next_to(band, RIGHT, buff=0.15)
        return VGroup(band, label)

    def build_all_paths(self, forward: bool = True) -> VGroup:
        """The walk as a solid: every step a beam standing in real space, grouped by step.

        Grouped so a beat can draw them column by column from the walk's own start. Drawn as
        lines this was a picture OF a walk; with thickness and depth it is an object, and two of
        them at different depths read as two objects rather than as one drawing twice.
        """
        front_z = 0.0 if forward else WALK_BACKWARD_Z
        centre_z = front_z - BEAM_DEPTH / 2
        colour = FORWARD_RAMP_COLOUR if forward else BACKWARD_RAMP_COLOUR
        beams = VGroup()
        for step in range(len(CATEGORIES)):
            column = VGroup()
            for net in range(-step, step + 1, 2):
                for rise in (1, -1):
                    start_x = self.walk_x(step if forward else len(CATEGORIES) - step)
                    end_x = self.walk_x(step + 1 if forward else len(CATEGORIES) - step - 1)
                    start_y, end_y = self.walk_y(net), self.walk_y(net + rise)
                    run, climb = end_x - start_x, end_y - start_y
                    beam = Prism(dimensions=[np.hypot(run, climb), BEAM_THICKNESS, BEAM_DEPTH])
                    beam.set_fill(colour, opacity=1.0)
                    beam.set_stroke(colour, width=0.5, opacity=0.5)
                    beam.rotate(np.arctan2(climb, run), axis=OUT)
                    beam.move_to([(start_x + end_x) / 2, (start_y + end_y) / 2, centre_z])
                    # Which category this step folds in. The forward walk's step s covers
                    # category s; the backward walk's step s covers the category that many
                    # places from the END. Carried on the beam so the cut can be decided by
                    # index rather than by comparing its midpoint against the plane -- the
                    # beams being EXCLUDED sit exactly on that boundary, so the comparison
                    # turned on floating-point luck, and for category zero the forward walk has
                    # to keep nothing at all.
                    beam.category = step if forward else len(CATEGORIES) - 1 - step
                    column.add(beam)
            beams.add(column)
        return anchor_walk_depth(
            beams, FORWARD_DEPTH_ANCHOR if forward else BACKWARD_DEPTH_ANCHOR, RAMP_LAYER)

    def build_zero_rail(self, forward: bool = True) -> VGroup:
        """The level line, flat on the ramp face and cut into one piece per step.

        Manim's 3D camera sorts whole mobjects by depth, so one bar across the scene takes ONE
        depth for its entire length and the beams nearer the camera draw over it -- the rail
        vanished about seventy percent along, ending in mid-air.
        """
        front_z = 0.0 if forward else WALK_BACKWARD_Z
        reach = (WALK_RIGHT_X - WALK_LEFT_X) / len(CATEGORIES)
        rail = VGroup()
        for step in range(len(CATEGORIES)):
            piece = Prism(dimensions=[reach, RAIL_THICKNESS, 0.012])
            piece.set_fill(RAIL_COLOUR, opacity=1.0).set_stroke(width=0)
            piece.move_to([WALK_LEFT_X + (step + 0.5) * reach, self.walk_y(0), front_z + 0.02])
            rail.add(piece)
        return anchor_walk_depth(
            rail, FORWARD_DEPTH_ANCHOR if forward else BACKWARD_DEPTH_ANCHOR, RAIL_LAYER)

    def build_cut_plane(self, category: int) -> Prism:
        """One slab standing across both ramps at a category, which is what a cut IS."""
        gap_x = (self.walk_x(category) + self.walk_x(category + 1)) / 2
        plane = Prism(dimensions=[0.05, len(CATEGORIES) * WALK_UNIT_Y * 2.4,
                                  abs(WALK_BACKWARD_Z) + 1.4])
        plane.set_fill(CUT_PLANE_COLOUR, opacity=0.30)
        plane.set_stroke(CUT_PLANE_COLOUR, width=1.5, opacity=0.8)
        plane.move_to([gap_x, self.walk_y(0), WALK_BACKWARD_Z / 2 + 0.2])
        # The slab reaches from behind the back walk to in front of the front one, so its own
        # midpoint puts it between them and half the cut disappears. Anchored in front instead,
        # which is what a sheet of glass over both of them should look like.
        return anchor_walk_depth(plane, CUT_DEPTH_ANCHOR, SPHERE_LAYER)

    def play_column_sweep(self, seconds: float) -> None:
        """Advance the distribution one category at a time, translating column into column.

        The whole point of the act is that this loop is the algorithm: one pass per category
        over a column of ten numbers, rather than a walk over 512 paths.
        """
        # The ramp stays, set back. It is the ground the distribution is measured over, and
        # taking it away would leave the spheres floating with nothing to be a summary OF.
        # Dimmed by opacity rather than by stroke: these are solid beams now, and a stroke
        # setting does nothing to a filled prism.
        dim_seconds = snap_to_frames(min(0.5, seconds * 0.12))
        self.play(self.all_paths.animate.set_opacity(WALK_DIMMED_RAMP_OPACITY),
                  run_time=dim_seconds)

        self.forward = [{0: 1.0}]
        for chance in WIN_CHANCES:
            self.forward.append(self.step_distribution(self.forward[-1], chance))

        # The backward table, built now and drawn later: suffix[i] is what categories i onward
        # contribute, so suffix[n] is the empty walk and suffix[0] is all nine.
        self.backward = [None] * (len(CATEGORIES) + 1)
        self.backward[len(CATEGORIES)] = {0: 1.0}
        for index in range(len(CATEGORIES) - 1, -1, -1):
            self.backward[index] = self.step_distribution(
                self.backward[index + 1], WIN_CHANCES[index])

        # A floor, because a sweep the viewer cannot follow is worse than one that runs a
        # little past its cue -- if the line ever gets too short, this overruns loudly instead.
        per_column = snap_to_frames(max(0.18, (seconds - dim_seconds) / len(self.forward)))
        self.forward_columns = Group()
        for step, distribution in enumerate(self.forward):
            column = self.build_column(distribution, step)
            self.forward_columns.add(column)
            self.play(FadeIn(column), run_time=per_column)

    def column_colour(self, net: int, forward: bool, majority_colour):
        """What one sphere is drawn in: its walk's colour, unless it is part of a highlight.

        `majority_colour` is only ever asked for on the forward walk's last column, and only
        the heights above the line belong to the majority it is highlighting.
        """
        if not forward:
            return GREEN_C
        if majority_colour is not None and net > 0:
            return majority_colour
        return BLUE_B

    def step_distribution(self, distribution: dict, chance: float) -> dict:
        """One category folded into a column: every height sends its mass up and down.

        This is the whole dynamic programme -- and it is used for BOTH sweeps, because a walk
        run backwards over the same independent categories obeys the same recurrence.
        """
        stepped = {}
        for net, probability in distribution.items():
            stepped[net + 1] = stepped.get(net + 1, 0.0) + probability * chance
            stepped[net - 1] = stepped.get(net - 1, 0.0) + probability * (1.0 - chance)
        return stepped

    def winning_mass(self, distribution: dict) -> float:
        """The probability of finishing above the line, which is the objective itself."""
        return sum(probability for net, probability in distribution.items() if net > 0)

    def build_column(self, distribution: dict, step: int, forward: bool = True,
                     majority_colour=None) -> VGroup:
        """One column of a sweep: a dot per reachable height, sized by its probability.

        The backward sweep sits on its own plane behind the forward one, in its own colour, so
        the tilted camera separates them in depth -- the cut in the next beat needs both to be
        readable at once, and side by side on one plane they overlapped.
        """
        depth = 0.0 if forward else WALK_BACKWARD_Z
        dots = Group()
        for net, probability in distribution.items():
            # Spheres, not discs. Flat circles have no thickness, so rotating the camera only
            # skewed a drawing; a sphere is the same shape from every angle and the tilt reveals
            # that the walk was a solid all along rather than a picture of one.
            dot = Dot3D(
                point=[self.walk_x(step), self.walk_y(net), depth],
                radius=SPHERE_MIN_RADIUS
                       + (SPHERE_MAX_RADIUS - SPHERE_MIN_RADIUS) * probability ** 0.5,
                resolution=(SPHERE_RESOLUTION, SPHERE_RESOLUTION),
                # ONE colour per walk. Size and opacity already carry the probability, and
                # splitting the colour by side of the line said the two halves were different
                # KINDS of thing when they are the same quantity either side of a line the rail
                # is already drawing.
                #
                # The one exception is the closing beat, which picks out the answer. It must
                # pick out the heights the answer is made OF -- net above zero, the same test
                # winning_mass sums and the same band build_win_region shades. This used to
                # colour the whole column, so the line "add those probabilities up" lit ten
                # spheres for a total that counts five, with the five below the line gold
                # inside a frame that also shaded them as NOT the majority.
                color=self.column_colour(net, forward, majority_colour),
            )
            # Remembered, not recomputed: a cut dims every column but the one it keeps, and the
            # column a previous cut kept has to come back to the opacity its own probability
            # earned rather than to a flat full strength.
            dot.base_opacity = 0.35 + 0.60 * probability ** 0.5
            # Which column this is: for the forward walk, how many categories have been folded
            # in; for the backward walk, the first category its suffix still covers.
            dot.column_index = step
            dots.add(dot.set_opacity(dot.base_opacity))
        return anchor_walk_depth(
            dots, FORWARD_DEPTH_ANCHOR if forward else BACKWARD_DEPTH_ANCHOR, SPHERE_LAYER)

    # ── Act four: two sweeps, and a cut that slides ───────────────────────────────────

    def play_tipping_point(self) -> None:
        """How the algorithm actually values a category: one forward sweep, one backward, and a
        gap slid along between them.

        This is what backend/math/algorithm_helpers.py does, not a restatement of it. It builds
        a prefix table sweeping one way and a suffix table sweeping the other, then for each
        category convolves the prefix that stops just before it with the suffix that starts just
        after -- `_leave_one_out_probability`, whose docstring is "P(the categories either side
        of the excluded one contribute exactly target_points)".

        The animation has to be the cheap version rather than the obvious one. Re-running an
        eight-step walk per category would look like the same answer and teach the opposite
        lesson: the point of the two tables is that nine categories cost two sweeps, not nine.
        """
        with self.voiceover(text=NARRATION['backward']):
            # Tilted only now. Everything before this is flat artwork that a rotated camera
            # would skew for no reason; the angle exists to separate the two sweeps in depth.
            self.move_camera(phi=WALK_VIEW_PHI * DEGREES,
                             theta=WALK_VIEW_THETA * DEGREES,
                             frame_center=TILTED_FRAME_CENTRE, run_time=1.6)

            # The second walk arrives as a second SOLID, drawn from its own start at the
            # right-hand end so its direction is visible rather than asserted.
            self.backward_ramp = self.build_all_paths(forward=False)
            self.backward_rail = self.build_zero_rail(forward=False)
            self.play(LaggedStart(*[Create(column) for column in self.backward_ramp],
                                  lag_ratio=WALK_DRAW_OVERLAP), run_time=2.2)
            self.play(FadeIn(self.backward_rail), run_time=0.5)
            self.backward_columns = Group()
            for step in range(len(CATEGORIES), -1, -1):
                column = self.build_column(self.backward[step], step, forward=False)
                self.backward_columns.add(column)
                self.play(FadeIn(column), run_time=0.30)
            self.wait(0.5)

        with self.voiceover(text=NARRATION['the_cut']) as tracker:
            wait_until_phrase(self, tracker, 'Take any of the categories')
            # The majority band goes with the walk it described. It belongs to the FORWARD walk
            # finishing above the line, which is not what this act is about, and left up it sat
            # over the cut claiming a region of a picture that no longer means that.
            self.play(FadeOut(self.win_region), run_time=0.5)
            self.draw_cut_marker(CUT_CATEGORY)
            self.cut_legend = self.build_cut_legend(CUT_CATEGORY)
            self.add_fixed_in_frame_mobjects(self.cut_legend)
            self.play(FadeIn(self.cut_legend), run_time=0.7)

            # The dimming IS the argument of this line, so it arrives in the two halves the line
            # names -- the walk that leads up to the cut, then the one that leads back to it --
            # instead of landing whole before the sentence has said what it means.
            # The material on the wrong side of the plane LEAVES. What survives is the two
            # categories the forward walk had taken by the cut and the six the backward walk
            # covers after it -- the other eight -- with the category being valued gone from
            # both, which is the whole of what this line claims.
            wait_until_phrase(self, tracker, 'went in opposite directions')
            _, self.cut_discarded = self.split_at_cut(CUT_CATEGORY)
            self.cut_travelled = self.glued_pieces(CUT_CATEGORY)
            self.play(FadeOut(self.cut_discarded), run_time=1.1)

            # The glue: the surviving suffix travels forward until its cut face meets the
            # prefix's, and the two become one walk running through eight categories.
            wait_until_phrase(self, tracker, 'are the opposite')
            self.play(self.cut_travelled.animate.shift([0, 0, -WALK_BACKWARD_Z]),
                      FadeOut(self.cut_marker), run_time=1.6)
            # Everything is back in the forward plane now, which rides higher on screen than the
            # pair did, so the camera follows it rather than leaving it drifting up the frame.
            self.move_camera(frame_center=GLUED_FRAME_CENTRE, run_time=0.8)

            wait_until_phrase(self, tracker, 'complete information')
            self.wait(0.6)

        with self.voiceover(text=NARRATION['convolution']) as tracker:
            # Drawn from nothing rather than created one arc at a time: adding the arcs
            # individually would leave the group itself outside the scene, and the slide's
            # Transform needs the group to be what is on screen.
            arcs, total = self.build_meeting_arcs(CUT_CATEGORY)
            for arc in arcs:
                arc.set_opacity(0.0)
            self.add(arcs)
            self.meeting_arcs = arcs

            # EVERY arc, on the words that name the convolution. The arcs ARE the
            # convolution, so the picture of it belongs on the phrase that introduces it.
            # Revealing them one clause at a time put a single hairline arc here and the ones
            # a viewer actually registers two and five seconds later, which reads as the
            # drawing lagging the narration however well the first one is anchored.
            # lead_seconds, not a bare anchor. Measured against the audio, the arcs used
            # to become visible 1.24 seconds after the words: the anchor started the fade on
            # the phrase, and a 0.8 second fade is still mostly transparent while the phrase
            # is being said. Leading by the fade's own length makes them FINISH arriving as
            # the convolution is named, which is what "draw them on these words" means.
            wait_until_phrase(self, tracker, 'check the convolutions', lead_seconds=0.8)
            self.play(*[arc.animate.set_opacity(arc.drawn_opacity) for arc in arcs],
                      run_time=0.8)

            wait_until_phrase(self, tracker, 'add up the products')
            self.meeting_readout = self.build_meeting_readout(CUT_CATEGORY, total)
            self.add_fixed_in_frame_mobjects(self.meeting_readout)
            self.play(FadeIn(self.meeting_readout), run_time=0.7)

        with self.voiceover(text=NARRATION['slide']) as tracker:
            # Each of these IS a fresh cut, because a cut takes material away and the next
            # category needs the material the last one discarded. Fast, though: the line's
            # whole claim is that doing it again is cheap.
            seconds = max(RECUT_MINIMUM_SECONDS,
                          (tracker.get_remaining_duration() - 1.2) / len(SLIDE_TO) - 1.1)
            for category in SLIDE_TO:
                self.recut_to(category, seconds)
                self.slide_meeting_to(category)

            # The scene change belongs to the END of this line rather than the start of the
            # next one. Flattening the camera and clearing the frame under the words 'This
            # gives us' cut that sentence in half; done here, the next line opens on a settled
            # frame and says its first words over the board it is about.
            self.wait(max(0.2, tracker.get_remaining_duration() - SCENE_CHANGE_SECONDS))
            # Back to flat for the closing board, which is 2D artwork like the opening.
            self.move_camera(phi=FLAT_VIEW_PHI * DEGREES,
                             theta=FLAT_VIEW_THETA * DEGREES, run_time=1.2)
            # Everything currently drawn, rather than a list of names that has to be kept in
            # step by hand -- the previous list had fallen behind and the board came up on top
            # of the leftover sweep.
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)
            self.clear()

        with self.voiceover(text=NARRATION['punting']) as tracker:
            board = self.build_punt_board()
            self.play(FadeIn(board['bars']), run_time=1.2)

            # The excluded category is actually taken off the board, not just described. The
            # whole claim is about what the OTHER eight look like, and that is a different
            # picture in each case rather than the same picture with different words over it.
            #
            # It lands on the clause that sets the claim up, so the board is already showing
            # eight categories by the time the word 'excluding' arrives -- anchored to that
            # word, the change was still happening while he explained its consequence.
            wait_until_phrase(self, tracker, 'high for the average category',
                              lead_seconds=0.8)
            self.play(*self.exclude_bar(board, AVERAGE_CATEGORY), run_time=0.8)

            wait_until_phrase(self, tracker, 'four-four tie with the others')
            self.play(FadeIn(board['contested']), run_time=0.9)
            self.wait(1.0)

            wait_until_phrase(self, tracker, 'For the other categories')
            self.play(*self.restore_bar(board, AVERAGE_CATEGORY),
                      FadeOut(board['contested']), run_time=0.5)
            self.play(*self.exclude_bar(board, EXCLUDED_DECISIVE), run_time=0.7)

            # All nine tipping points, here, rather than a second verdict card followed by
            # the strip ten seconds later. The card said 'take away a decisive category, the
            # rest sit 5 - 3' -- words about a board state that was never drawn, while the one
            # picture that answers the line directly waited until the closing sentence. The
            # strip arrives on the clause it belongs to and holds for the rest of the act.
            wait_until_phrase(self, tracker, 'already likely have won or lost')
            self.play(*self.restore_bar(board, EXCLUDED_DECISIVE), run_time=0.5)
            self.play(FadeIn(self.build_tipping_strip()), run_time=1.0)
            self.wait(max(0.5, tracker.get_remaining_duration() - 0.4))
        self.wait(0.6)

    def board_tipping_points(self) -> list:
        """Each category's tipping point on the BOARD's team, counted over all four-four splits.

        The board's own odds, not the walk's: four categories won decisively, four lost
        decisively, one a coin flip. The whole claim depends on that being extreme, since at
        realistic odds the nine tipping points differ by less than two points and the argument
        has nothing to show. Here the average category comes to 69.3% against 43.5% for every
        decisive one, won or lost alike.
        """
        chances = PUNT_BOARD_ODDS
        points = []
        for index in range(len(chances)):
            others = chances[:index] + chances[index + 1:]
            total = 0.0
            for number in range(2 ** len(others)):
                bits = [(number >> shift) & 1 for shift in range(len(others) - 1, -1, -1)]
                if sum(bits) != MAJORITY - 1:
                    continue
                chance = 1.0
                for bit, probability in zip(bits, others):
                    chance *= probability if bit else (1.0 - probability)
                total += chance
            points.append(total)
        return points

    def build_tipping_strip(self) -> VGroup:
        """A bar under each category, as tall as that category's tipping point.

        The board above says which categories this team contests; this says what the algorithm
        gets for contesting them. Drawn to a shared scale so the two groups can be compared by
        height alone -- which is the entire point, and is why the numbers are written on as
        well: the gap is large but a viewer should not have to take the drawing's word for it.
        """
        points = self.board_tipping_points()
        tallest = max(points)
        strip = VGroup()
        for index, value in enumerate(points):
            height = TIPPING_STRIP_HEIGHT * value / tallest
            x = -4.6 + index * 1.15
            bar = Rectangle(width=0.5, height=height, stroke_width=0, fill_opacity=1.0,
                            fill_color=TIPPING_STRIP_COLOUR)
            bar.move_to([x, TIPPING_STRIP_BASE + height / 2, 0.0])
            strip.add(bar)
            strip.add(Text(f'{value:.0%}', font_size=17, color=TIPPING_STRIP_COLOUR)
                      .move_to([x, TIPPING_STRIP_BASE - 0.26, 0.0]))
        caption = Text('tipping point probability', font_size=21,
                       color=TIPPING_STRIP_COLOUR)
        caption.move_to([0.0, TIPPING_STRIP_BASE + TIPPING_STRIP_HEIGHT + 0.28, 0.0])
        strip.add(caption)
        return strip

    def exclude_bar(self, board, index: int):
        """Take one category off the board: its level greyed out and struck through.

        Only the FILL is dimmed. Dimming the whole bar also touches its outline, whose fill is
        deliberately empty -- restoring that to full opacity painted a white block across the
        top of the bar.
        """
        outline, filled = board['bars'][1][index]
        return [filled.animate.set_fill(GREY_D, opacity=0.25),
                outline.animate.set_stroke(GREY_D, opacity=0.35),
                board['bars'][2][index].animate.set_opacity(0.25),
                FadeIn(board['strikes'][index])]

    def restore_bar(self, board, index: int):
        outline, filled = board['bars'][1][index]
        # Back to the colour the bar was BUILT in, which now has three cases -- restoring
        # the average category to the losing grey would quietly recolour the one bar the
        # closing argument is about.
        odds = PUNT_BOARD_ODDS[index]
        colour = BLUE_D if odds > 0.5 else (GREY_D if odds < 0.5 else GREY_B)
        return [filled.animate.set_fill(colour, opacity=0.9),
                outline.animate.set_stroke(GREY_D, opacity=1.0),
                board['bars'][2][index].animate.set_opacity(1.0),
                FadeOut(board['strikes'][index])]

    def build_punt_board(self) -> dict:
        """Four categories won decisively, four lost, one average, and what removing one does.

        The two cases are the whole punting argument and they differ by one column. Drop the
        AVERAGE category and the remaining eight split four-four: level, so the one dropped
        decides the matchup. Drop a decisive one and the rest are usually already settled, so
        it decides much less. Same board, opposite conclusions.
        """
        bars = VGroup()
        for index in range(len(CATEGORIES)):
            odds = PUNT_BOARD_ODDS[index]
            x = -4.6 + index * 1.15
            outline = Rectangle(width=0.78, height=PUNT_BAR_HEIGHT, stroke_width=2,
                                stroke_color=GREY_D, fill_opacity=0.0)
            outline.move_to([x, 0.9, 0.0])
            filled_height = PUNT_BAR_HEIGHT * odds
            # Three states now, not two: won, lost, and the one in between. The average
            # category gets its own colour because the whole closing argument turns on it
            # being neither of the other two.
            fill_colour = BLUE_D if odds > 0.5 else (GREY_D if odds < 0.5 else GREY_B)
            filled = Rectangle(width=0.78, height=filled_height, stroke_width=0,
                               fill_color=fill_colour, fill_opacity=0.9)
            filled.move_to([x, 0.9 - PUNT_BAR_HEIGHT / 2 + filled_height / 2, 0.0])
            bars.add(VGroup(outline, filled))
        # In the words the narration uses. 'Held' and 'given up' were a private vocabulary for
        # what the line calls contesting and punting, and a board should not need translating.
        readouts = VGroup(*[
            Text(f'{PUNT_BOARD_ODDS[index]:.0%}', font_size=17, color=GREY_B)
            .move_to([-4.6 + index * 1.15, -0.32, 0.0])
            for index in range(len(CATEGORIES))
        ])
        # Above the board, not below it: the tipping-point strip's tallest bar reaches
        # y = -0.70 and a caption under the board would sit on top of it.
        caption = Text('category win rate', font_size=21, color=GREY_B)
        caption.move_to([0.0, 2.35, 0.0])

        def verdict(title, split, note, colour, y):
            return VGroup(
                Text(title, font_size=23, color=colour),
                Text(split, font_size=27, color=colour),
                Text(note, font_size=22, color=GREY_B),
            ).arrange(RIGHT, buff=0.5).move_to([0.0, y, 0.0])

        strikes = VGroup(*[
            Line([-4.6 + index * 1.15 - 0.45, 0.9, 0.0],
                 [-4.6 + index * 1.15 + 0.45, 0.9, 0.0],
                 color=GREY_B, stroke_width=4)
            for index in range(len(CATEGORIES))
        ])
        return {
            'bars': VGroup(caption, bars, readouts),
            'strikes': strikes,
            'contested': verdict('take away the average category', 'the rest sit 4 - 4',
                                 'level, so it decides the matchup', YELLOW, -1.1),
        }

    def play_the_slope(self) -> None:
        """The question the next two acts answer, asked where the script asks it.

        This ran AFTER the tipping-point explanation, which put the answer before the question:
        the script asks how you differentiate the thing just built, and only then says what the
        derivative turns out to be.
        """
        with self.voiceover(text=NARRATION['tipping']) as tracker:
            # This line asks how you differentiate the thing just built, and had been left as a
            # bare wait -- the one beat in the scene with nothing to look at. The answer it is
            # reaching for is a slope, so the question is put as one: nudge a category, and ask
            # how far the total above the line moves.
            wait_until_phrase(self, tracker, 'calculate the slope')
            question = VGroup(
                Text('if one category gets better...', font_size=26, color=GREY_B),
                Text('...how much does the majority chance move?', font_size=26, color=YELLOW),
            ).arrange(DOWN, buff=0.34).move_to([0.0, self.walk_y(9) + 1.25, 0.0])
            self.play(FadeIn(question[0]), run_time=0.8)
            self.wait(0.6)
            self.play(FadeIn(question[1]), run_time=0.8)
            self.wait(max(0.5, tracker.get_remaining_duration() - 1.0))
            self.play(FadeOut(question), run_time=0.6)

    def play_what_tipping_means(self) -> None:
        """Why the gradient is a probability: the other eight settle it unless they are level.

        Eight toggles for the other categories, the chance of taking the one in question, and
        the objective beside them. Flip the toggles and the objective is 1 or 0 whatever that
        chance is -- the matchup is already decided. Set them level and the objective becomes
        the chance itself. So the slope is the probability of landing in that middle case, and
        the scenario table is that probability written out: the level rows are the only ones
        that respond, and every other row is flat.
        """
        with self.voiceover(text=NARRATION['tipping_point_probability']) as tracker:
            # The majority total goes with the walk it belongs to. Pinned in screen space, it
            # was not in the group being cleared and sat over this act as a stray number.
            self.play(FadeOut(Group(self.forward_columns, self.all_paths,
                                    self.lattice, self.win_region)),
                      FadeOut(self.majority_readout), run_time=0.7)
            panel = self.build_tipping_panel()
            # Both readouts have to be ADDED, not only transformed later: a Transform on a
            # mobject the scene never received shows nothing, which is how the objective column
            # came to be missing entirely.
            self.play(FadeIn(panel['frame']), FadeIn(panel['objective']),
                      FadeIn(panel['gradient']), run_time=1.0)

            # The level case, set on the clause that introduces it so the board is already
            # even by the time the words 'precisely even' arrive.
            wait_until_phrase(self, tracker, 'If the other eight categories are precisely tied')
            self.balance = self.build_balance(LEVEL_CASE)
            self.play(*self.set_toggles(panel, LEVEL_CASE),
                      Transform(panel['objective'], self.objective_readout(LEVEL_CASE)),
                      Transform(panel['gradient'], self.gradient_readout(LEVEL_CASE)),
                      FadeIn(self.balance), run_time=0.45)
            self.wait(0.9)

            # Several different even boards, all landing on the same pair of numbers. That is
            # what makes the answer a probability rather than a special case.
            for others in OTHER_LEVEL_CASES[:2]:
                self.play(*self.set_toggles(panel, others),
                          Transform(panel['objective'], self.objective_readout(others)),
                          Transform(panel['gradient'], self.gradient_readout(others)),
                          Transform(self.balance, self.build_balance(others)),
                          run_time=0.45)
                self.wait(0.8)

            equation = self.build_gradient_equation()
            wait_until_phrase(self, tracker, 'The gradient is exactly one')
            self.play(FadeIn(equation[0]), run_time=0.6)

            wait_until_phrase(self, tracker, 'On the other hand')
            # The beam tips here, hard, and that is the whole argument for the name: a board
            # that is already decided is one the ninth category cannot move.
            for others in (DECIDED_WON, DECIDED_LOST):
                self.play(*self.set_toggles(panel, others),
                          Transform(panel['objective'], self.objective_readout(others)),
                          Transform(panel['gradient'], self.gradient_readout(others)),
                          Transform(self.balance, self.build_balance(others)),
                          run_time=0.5)
                self.wait(1.6)

            wait_until_phrase(self, tracker, 'the gradient is zero')
            self.play(FadeIn(equation[1]), run_time=0.6)

            wait_until_phrase(self, tracker, 'That means the gradient')
            self.play(FadeIn(equation[2]), run_time=0.7)

            wait_until_phrase(self, tracker, 'we call this a tipping point')
            total = self.level_probability()
            summary = VGroup(
                Text(f'the other eight land tied {total:.1%} of the time',
                     font_size=26, color=YELLOW),
                Text('that is the tipping point probability', font_size=22, color=GREY_B),
            ).arrange(DOWN, buff=0.24).move_to([0.0, EQUATION_Y - 1.05, 0.0])
            self.play(FadeIn(summary), run_time=0.7)
            self.wait(max(0.5, tracker.get_remaining_duration() - 0.6))

            # Handed back. This act borrows the frame from the walk, and the cut that follows
            # reaches for the forward sweep and the lattice again -- left cleared, the cut would
            # open on a backward sweep with nothing to meet.
            self.play(FadeOut(panel['frame']), FadeOut(panel['objective']),
                      FadeOut(panel['gradient']), FadeOut(equation),
                      FadeOut(self.balance), FadeOut(summary), run_time=0.5)
            self.play(FadeIn(Group(self.lattice, self.all_paths, self.win_region,
                                   self.forward_columns)), run_time=0.7)

    def build_tipping_panel(self) -> dict:
        """All nine categories in one row, the focus among them rather than set apart.

        The eight used to sit in a block with the ninth off to the side under its own heading,
        which made the focus look like a different KIND of thing. It is not -- any of the nine
        could be the category being evaluated, and the argument only works because they are
        interchangeable. So they are drawn identically, in one row, and the one in question is
        simply the middle one, marked.
        """
        self.toggles = VGroup()
        cells = VGroup()
        for index in range(len(CATEGORIES)):
            x = -(len(CATEGORIES) - 1) / 2 * PANEL_CELL_GAP + index * PANEL_CELL_GAP
            focus = index == FOCUS_INDEX
            cell = Rectangle(
                width=PANEL_CELL_WIDTH, height=PANEL_CELL_HEIGHT, stroke_width=2,
                stroke_color=YELLOW if focus else GREY_D,
                fill_color=GREY_D, fill_opacity=0.0 if focus else 0.25,
            ).move_to([x, PANEL_ROW_Y, 0.0])
            cells.add(cell)
            if focus:
                cells.add(Text('p', font_size=32, color=YELLOW).move_to(cell.get_center()))
            else:
                self.toggles.add(cell)

        # No heading over the row. It read 'nine categories, any one of them', which is a
        # caption explaining a picture that already says it: nine identical cells with one of
        # them marked. The marker under the row is the only label the panel needs.
        marker = Text('the category being evaluated', font_size=19, color=YELLOW)
        marker.move_to([0.0, PANEL_ROW_Y - 0.62, 0.0])
        return {'frame': VGroup(cells, marker),
                'objective': self.objective_readout(None),
                'gradient': self.gradient_readout(None)}

    def build_balance(self, others) -> VGroup:
        """The other eight on a beam, tilted by how far from level they are.

        Level is not a special case of this picture, it IS the picture: a balance that happens
        to be level is the only state in which one more weight decides which way it goes. That
        is what the category being evaluated does, and what makes its influence a probability
        -- the chance of finding the beam balanced when it arrives.
        """
        won = sum(others) if others is not None else 0
        lost = (len(others) - won) if others is not None else 0
        share = 0.0 if others is None else (won - lost) / len(others)
        # Negated, and it has to be. The wins pan hangs off the +x end, and rotating a beam by
        # a POSITIVE angle lifts that end -- so tilting by the share directly sent the heavier
        # side up. Six losses against two wins raised the six, which is the opposite of what a
        # balance does and undercuts the only thing the picture is there to say.
        tilt = -share * BALANCE_MAX_TILT * DEGREES

        beam = Rectangle(width=BALANCE_WIDTH, height=0.075, stroke_width=0,
                         fill_opacity=1.0, fill_color=GREY_B)
        beam.rotate(tilt).move_to([0.0, BALANCE_Y, 0.0])

        reach = BALANCE_WIDTH / 2
        ends = VGroup()
        # The same two colours a loss and a win are drawn in everywhere else in the scene:
        # BLUE_D filled for a category taken, GREY_D for one lost. The pans were light grey,
        # which is the colour of the beam and of nothing else -- so the losing side read as
        # part of the apparatus rather than as the losses it is weighing.
        for side, count, pan_colour, text_colour in ((-1, lost, GREY_D, GREY_B),
                                                     (1, won, BLUE_D, BLUE_B)):
            x = side * reach * np.cos(tilt)
            y = BALANCE_Y + side * reach * np.sin(tilt)
            pan = Rectangle(width=0.5, height=0.14, stroke_width=0, fill_opacity=1.0,
                            fill_color=pan_colour).move_to([x, y - 0.15, 0.0])
            ends.add(pan)
            ends.add(Text(str(count), font_size=24, color=text_colour)
                     .move_to([x, y + 0.3, 0.0]))

        # A fulcrum, so the beam reads as something that can tip rather than a floating line.
        post = Rectangle(width=0.09, height=0.42, stroke_width=0, fill_opacity=1.0,
                         fill_color=GREY_D).move_to([0.0, BALANCE_Y - 0.28, 0.0])
        return VGroup(post, beam, ends)

    def objective_readout(self, others) -> VGroup:
        """What the majority objective comes to, given what the other eight did."""
        if others is None:
            value = '?'
        else:
            won = sum(others)
            value = 'p' if won == MAJORITY - 1 else ('1' if won >= MAJORITY else '0')
        return VGroup(
            Text('chance of the majority', font_size=22, color=GREY_B),
            Text(value, font_size=46, color=YELLOW if value == 'p' else GREY_B),
        ).arrange(DOWN, buff=0.22).move_to([CHANCE_READOUT_X, READOUT_ROW_Y, 0.0])

    def gradient_readout(self, others) -> VGroup:
        """How much the objective moves with p -- one when the rest are level, zero otherwise.

        Its own readout, beside the objective rather than instead of it. The two are different
        quantities: a decided board has a majority chance of 1 or 0 and a gradient of zero
        either way, so a single number on screen was bound to contradict whichever of them the
        line happened to be talking about.
        """
        if others is None:
            value = '?'
        else:
            value = '1' if sum(others) == MAJORITY - 1 else '0'
        return VGroup(
            Text('the gradient', font_size=22, color=GREY_B),
            Text(value, font_size=46, color=YELLOW if value == '1' else GREY_B),
        ).arrange(DOWN, buff=0.22).move_to([GRADIENT_READOUT_X, READOUT_ROW_Y, 0.0])

    def build_gradient_equation(self) -> VGroup:
        """The identity the whole act is for, in the three pieces the line states it in.

        Level contributes its own probability times a gradient of one; everything else
        contributes its probability times zero; so the gradient of the whole function is just
        the chance of being level. Assembled term by term as each is spoken rather than
        arriving whole.
        """
        pieces = VGroup(
            MathTex(r'P(\text{tied}) \times 1', font_size=EQUATION_FONT, color=YELLOW),
            MathTex(r'+\; P(\text{not tied}) \times 0', font_size=EQUATION_FONT, color=GREY_B),
            MathTex(r'=\; P(\text{tied})', font_size=EQUATION_FONT, color=YELLOW),
        ).arrange(RIGHT, buff=0.26).move_to([0.0, EQUATION_Y, 0.0])
        return pieces

    def set_toggles(self, panel, others):
        """Light the toggles for a given outcome of the other eight."""
        return [
            toggle.animate.set_fill(BLUE_D if won else GREY_D,
                                    opacity=0.9 if won else 0.25)
            for toggle, won in zip(self.toggles, others)
        ]

    def level_probability(self) -> float:
        """How often the other eight land level -- the tipping point probability itself.

        Summed directly over every four-four split of eight independent categories. The act
        that follows computes the same quantity by convolving a forward sweep with a backward
        one, so the two halves of the scene can be checked against each other.
        """
        # The other eight are the eight that are NOT the one the panel marks. This counted the
        # first eight instead -- excluding the LAST category while the picture pointed at the
        # middle one -- so the figure on screen belonged to a different category than the one
        # being valued: 29.2% where the answer is 54.7%.
        others = WIN_CHANCES[:FOCUS_INDEX] + WIN_CHANCES[FOCUS_INDEX + 1:]
        total = 0.0
        for number in range(2 ** len(others)):
            bits = [(number >> shift) & 1 for shift in range(len(others) - 1, -1, -1)]
            if sum(bits) != MAJORITY - 1:
                continue
            chance = 1.0
            for bit, probability in zip(bits, others):
                chance *= probability if bit else (1.0 - probability)
            total += chance
        return total

    def build_cut_marker(self, category: int) -> tuple:
        """The dashed gap at one category, and the category's name pinned above the frame."""
        gap_x = (self.walk_x(category) + self.walk_x(category + 1)) / 2
        marker = self.build_cut_plane(category)
        # Above the frame rather than below it: the readouts already own the bottom, and at this
        # camera angle the lower half of the screen is where the backward sweep sits.
        label = Text(CATEGORIES[category], font_size=21, color=YELLOW)
        label.move_to([0.0, CUT_LABEL_SCREEN_Y, 0.0])
        return marker, label

    def build_cut_legend(self, category: int) -> VGroup:
        """The nine categories, each in the colour of the walk that accounts for it.

        The cut divides them three ways and the ramps alone cannot say which is which: the ones
        before it belong to the forward walk, the ones after it to the backward walk, and the
        one being valued belongs to neither -- which is the point, and the only one of the three
        that a viewer cannot read off the picture.
        """
        entries = VGroup()
        for index, name in enumerate(CATEGORIES):
            colour = BLUE_B if index < category else (
                YELLOW if index == category else GREEN_C)
            entries.add(Text(name, font_size=17, color=colour))
        entries.arrange(DOWN, buff=CUT_LEGEND_GAP - 0.18, aligned_edge=LEFT)
        return entries.move_to([CUT_LEGEND_X, 0.0, 0.0])

    def recut_to(self, category: int, seconds: float) -> None:
        """Put the two walks back as they were, then cut and glue them somewhere else.

        The cut REMOVES material, so a slide cannot just move a readout: the columns another
        category needs were thrown away by the cut before it. Drawing its arcs anyway put them
        across empty space -- sliding to the sixth category asked for a forward column that had
        been discarded, and sliding to the first asked for a backward one.

        So each new cut is a real cut: the discarded half comes back, the glued half returns to
        its own depth, and the whole thing happens again at the new category. Quickly, because
        the line says this is the cheap part.
        """
        # The old answer goes FIRST. Left up while the label and legend moved to the new
        # category, the frame showed one category named above and another's number below --
        # for the second or so it took the arcs to catch up.
        self.play(FadeIn(self.cut_discarded),
                  FadeOut(self.meeting_readout), FadeOut(self.meeting_arcs),
                  self.cut_travelled.animate.shift([0, 0, WALK_BACKWARD_Z]),
                  run_time=seconds * 0.4)

        marker, label = self.build_cut_marker(category)
        legend = self.build_cut_legend(category)
        self.add_fixed_in_frame_mobjects(label, legend)
        self.play(FadeOut(self.cut_label), FadeOut(self.cut_legend),
                  FadeIn(label), FadeIn(legend), FadeIn(marker), run_time=seconds * 0.25)
        self.cut_marker, self.cut_label, self.cut_legend = marker, label, legend

        _, self.cut_discarded = self.split_at_cut(category)
        self.cut_travelled = self.glued_pieces(category)
        self.play(FadeOut(self.cut_discarded), FadeOut(self.cut_marker),
                  self.cut_travelled.animate.shift([0, 0, -WALK_BACKWARD_Z]),
                  run_time=seconds * 0.35)

    def split_at_cut(self, category: int):
        """Everything the cut keeps and everything it throws away, on both walks.

        Sorted by which side of the plane a piece sits on, which is what makes the exclusion
        real: the beams for the category BEING VALUED span the gap the plane stands in, so they
        fall on neither side and leave with the discarded material. Cutting the third category
        keeps two steps of the forward walk and six of the backward one -- the other eight --
        and category three itself is in neither piece.

        Dimming a column said the same thing and showed the opposite: every beam stayed drawn,
        so the walk still ran through the category the argument had just excluded.
        """
        kept, gone = Group(), Group()
        for ramp, forward in ((self.all_paths, True), (self.backward_ramp, False)):
            for column in ramp:
                for beam in column:
                    covered = (beam.category < category if forward
                               else beam.category > category)
                    (kept if covered else gone).add(beam)
        for columns, forward in ((self.forward_columns, True),
                                 (self.backward_columns, False)):
            for column in columns:
                for dot in column:
                    covered = (dot.column_index <= category if forward
                               else dot.column_index > category)
                    (kept if covered else gone).add(dot)
        # The rails have no category of their own, so they go by position -- and a rail segment
        # never straddles the plane, so there is no boundary case to get wrong.
        boundary = (self.walk_x(category) + self.walk_x(category + 1)) / 2
        for rail, keep_left in ((self.lattice[0], True), (self.backward_rail, False)):
            for piece in rail:
                side = kept if (piece.get_center()[0] < boundary) == keep_left else gone
                side.add(piece)
        return kept, gone

    def glued_pieces(self, category: int) -> Group:
        """The half of the cut that has to travel: everything the BACKWARD walk keeps."""
        travelling = Group()
        for column in self.backward_ramp:
            for beam in column:
                if beam.category > category:
                    travelling.add(beam)
        for column in self.backward_columns:
            for dot in column:
                if dot.column_index > category:
                    travelling.add(dot)
        boundary = (self.walk_x(category) + self.walk_x(category + 1)) / 2
        for piece in self.backward_rail:
            if piece.get_center()[0] >= boundary:
                travelling.add(piece)
        return travelling

    def emphasise_single_column(self, columns, kept: int) -> list:
        """Bring one column of a sweep back to full strength and push every other one down.

        Every column is given an explicit opacity rather than only the unwanted ones being
        faded. The column a previous cut kept was skipped by that fade and never restored, so
        from the second cut onwards the whole picture was dim.
        """
        return [
            dot.animate.set_opacity(dot.base_opacity if index == kept else DIMMED_OPACITY)
            for index, column in enumerate(columns) for dot in column
        ]

    def backward_column_index(self, category: int) -> int:
        """Where the suffix that starts after a category sits, counting from the last sweep."""
        return len(CATEGORIES) - (category + 1)

    def draw_cut_marker(self, category: int) -> None:
        """Open the gap at a category before anything has been dimmed for it."""
        self.cut_marker, self.cut_label = self.build_cut_marker(category)
        self.add_fixed_in_frame_mobjects(self.cut_label)
        self.play(Create(self.cut_marker), FadeIn(self.cut_label), run_time=1.0)

    def show_cut_at(self, category: int) -> None:
        """Move the gap to another category, keeping only what reaches it from either side."""
        marker, label = self.build_cut_marker(category)
        self.play(Transform(self.cut_marker, marker), FadeOut(self.cut_label),
                  *self.emphasise_single_column(self.forward_columns, category),
                  *self.emphasise_single_column(self.backward_columns,
                                                self.backward_column_index(category)),
                  run_time=0.6)
        self.add_fixed_in_frame_mobjects(label)
        self.play(FadeIn(label), run_time=0.3)
        self.cut_label = label

    def build_meeting_arcs(self, category: int) -> tuple:
        """Pair each height on the left with the opposite height on the right, and total it.

        A height of plus h before the category and minus h after it sum to level, which is the
        only way the category can be the one that decides the matchup. Ordered heaviest pair
        first, so a beat that draws them one at a time leads with the one that carries the most.
        """
        before, after = self.forward[category], self.backward[category + 1]
        weighted_heights = []
        for height, probability in before.items():
            partner = after.get(-height)
            if partner is None:
                continue
            weighted_heights.append((probability * partner, height))
        weighted_heights.sort(reverse=True)

        arcs = VGroup()
        for weight, height in weighted_heights:
            arc = Line(
                [self.walk_x(category), self.walk_y(height), 0.0],
                # The suffix has been glued forward, so its heights are in the forward plane.
                [self.walk_x(category + 1), self.walk_y(-height), 0.0],
                color=YELLOW, stroke_width=1.0 + 7.0 * weight ** 0.5,
            )
            arc.drawn_opacity = 0.35 + 0.65 * weight ** 0.5
            arcs.add(arc.set_opacity(arc.drawn_opacity))
        return arcs, sum(weight for weight, _ in weighted_heights)

    def build_meeting_readout(self, category: int, total: float) -> Text:
        """The convolution's answer, pinned to the screen rather than to the tilted walk."""
        readout = Text(f'{CATEGORIES[category]} is decisive {total:.1%} of the time',
                       font_size=26, color=YELLOW)
        return readout.move_to([0.0, READOUT_SCREEN_Y, 0.0])

    def slide_meeting_to(self, category: int) -> None:
        """Pair the heights at this cut, and say what they come to.

        Both arrive fresh rather than morphing out of the last cut's pair. The readout is
        REPLACED because Transform morphs one Text into another glyph by glyph, and between two
        lines of different length it drags letters across the frame and strands the leftovers.
        The arcs are replaced for the same reason in reverse: two cuts rarely have the same
        number of pairings, and transforming three arcs into five gave the extra two nowhere
        to come from.
        """
        arcs, total = self.build_meeting_arcs(category)
        readout = self.build_meeting_readout(category, total)
        self.meeting_arcs, self.meeting_readout = arcs, readout
        self.add_fixed_in_frame_mobjects(readout)
        self.play(FadeIn(arcs), FadeIn(readout), run_time=0.8)
