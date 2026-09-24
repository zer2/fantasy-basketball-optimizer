"""Bake the enumeration of winning scenarios to one image.

The table is a still picture that only ever translates, but Manim redraws every mobject on every
frame whether it is in shot or not -- so scrolling it as eight hundred vector objects cost 170
seconds of a 258 second render, in one animation. As an image it costs one object a frame.

Run from the repository root whenever WIN_CHANCES, the row count or the cell layout changes:

    python visualizations/wireframes/most_categories/prepare_outcome_table.py

The scene fails loudly if the image is missing or was baked from different numbers, so a stale
image cannot quietly survive an edit to the odds.
"""
from __future__ import annotations

import json
import pathlib
import sys

from manim import Camera, tempconfig, BLACK, ORIGIN

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from most_categories import (                                          # noqa: E402
    build_outcome_table, TABLE_ROWS_BUILT, TABLE_ROW_GAP, WIN_CHANCES,
)

HERE = pathlib.Path(__file__).resolve().parent
IMAGE_PATH = HERE / 'outcome_table.png'
RECORD_PATH = HERE / 'outcome_table.json'
# Enough that a row is as crisp in the image as it would have been drawn. The scene shows about
# ten rows across five and a half units of an eight unit frame, so at 1080p a row is near
# seventy pixels; this gives it the same.
PIXELS_PER_UNIT = 140
MARGIN = 0.12


def main() -> None:
    table = build_outcome_table()
    table.move_to(ORIGIN)
    width, height = table.width + MARGIN, table.height + MARGIN

    with tempconfig({
        'frame_width': width,
        'frame_height': height,
        'pixel_width': int(round(width * PIXELS_PER_UNIT)),
        'pixel_height': int(round(height * PIXELS_PER_UNIT)),
        'background_color': BLACK,
    }):
        camera = Camera()
        camera.capture_mobjects([table])
        camera.get_image().save(IMAGE_PATH)

    RECORD_PATH.write_text(json.dumps({
        'rows': TABLE_ROWS_BUILT,
        'row_gap': TABLE_ROW_GAP,
        'win_chances': list(WIN_CHANCES),
        'scene_width': width,
        'scene_height': height,
    }, indent=2), encoding='utf-8')

    print(f'wrote {IMAGE_PATH.name}: {int(width * PIXELS_PER_UNIT)} x '
          f'{int(height * PIXELS_PER_UNIT)} pixels, {width:.2f} x {height:.2f} scene units')
    print(f'      {TABLE_ROWS_BUILT} winning scenarios at odds {WIN_CHANCES}')


if __name__ == '__main__':
    main()
