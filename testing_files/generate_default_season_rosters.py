# testing_files/generate_default_season_rosters.py
# Regenerates the default Season Mode rosters by running an H-score snake draft
# against the 2025-26 historical data.
#
# Process: 12 drafters, 13 picks, EC scoring, snake order. For each pick, run
# evaluate from the current drafter's perspective on the current board state and
# select the top H-score candidate.
#
# Emits both targets:
#   - frontend/data_entry/season/default_season_rosters.ts (TS export)
#   - testing_files/test_and_benchmark_season.py (Python literal — printed for manual paste)

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # repo root, so `backend` and `testing_files` resolve
sys.path.insert(0, str(_HERE))          # so `benchmark_helpers` resolves as a top-level module

from benchmark_helpers import client, _build_session_request
from backend.session import get_session
from backend.evaluate import run_evaluate

SEASON      = '2025-26'
N_DRAFTERS  = 12
N_PICKS     = 13


def generate_snake_draft_rosters() -> dict[str, list[str]]:
    """Run an H-score snake draft on an empty board for SEASON, returning rosters by drafter."""
    session_request = _build_session_request(scoring_format='Head to Head: Each Category')
    session_request['data_source']['season'] = SEASON

    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id   = response.json()['session_id']
    session      = get_session(session_id)
    n_iterations = session.current_params['n_iterations']

    team_names    = [f'Drafter {i + 1}' for i in range(N_DRAFTERS)]
    assignments   = {name: [] for name in team_names}

    for pick_row in range(N_PICKS):
        for slot in range(N_DRAFTERS):
            # Serpentine: even rounds go 0..N-1, odd rounds go N-1..0.
            drafter_index = slot if pick_row % 2 == 0 else (N_DRAFTERS - 1 - slot)
            drafter_name  = team_names[drafter_index]

            result = run_evaluate(
                session            = session
                , player_assignments = assignments
                , my_team_id         = drafter_name
                , exclusion_list     = []
                , remaining_cash     = None
            )
            assert result.candidates, f'No candidates returned for {drafter_name} at pick {pick_row + 1}'
            top_player = result.candidates[0].name
            assignments[drafter_name].append(top_player)
            print(f'  Round {pick_row + 1:2d}, {drafter_name}: {top_player}', flush=True)

    return assignments


def format_typescript(rosters: dict[str, list[str]]) -> str:
    """Render the TS DEFAULT_SEASON_ROSTERS export."""
    lines: list[str] = []
    lines.append('// default_season_rosters.ts')
    lines.append(f'// Pre-computed default Season Mode rosters: EC scoring, {SEASON} historical data,')
    lines.append(f'// {N_DRAFTERS} drafters, {N_PICKS} picks, snake-drafted by H-score rank.')
    lines.append('// Used to pre-fill the Season Mode roster table consistently across the app and tests.')
    lines.append('')
    lines.append('export const DEFAULT_SEASON_ROSTERS: Record<string, string[]> = {')
    for team, players in rosters.items():
        lines.append(f"    '{team}': [")
        for player in players:
            escaped = player.replace("'", "\\'") if "'" in player else player
            quote = '"' if "'" in player and '"' not in player else "'"
            if quote == "'":
                player_literal = f"'{escaped}'"
            else:
                player_literal = f'"{player}"'
            lines.append(f'        {player_literal},')
        lines.append('    ],')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


def format_python_literal(rosters: dict[str, list[str]]) -> str:
    """Render the Python dict literal for _DEFAULT_SEASON_ROSTERS in test_and_benchmark_season.py."""
    lines: list[str] = []
    lines.append('_DEFAULT_SEASON_ROSTERS: dict[str, list[str]] = {')
    for team, players in rosters.items():
        lines.append(f"    '{team}': [")
        for player in players:
            if "'" in player and '"' not in player:
                lines.append(f'        "{player}",')
            else:
                escaped = player.replace("'", "\\'")
                lines.append(f"        '{escaped}',")
        lines.append('    ],')
    lines.append('}')
    return '\n'.join(lines)


def main() -> None:
    print(f'Generating default season rosters for {SEASON} ...', flush=True)
    rosters = generate_snake_draft_rosters()

    ts_path = Path(__file__).parent.parent / 'frontend' / 'data_entry' / 'season' / 'default_season_rosters.ts'
    ts_path.write_text(format_typescript(rosters), encoding='utf-8')
    print(f'\nWrote {ts_path}', flush=True)

    print('\n--- Python literal for testing_files/test_and_benchmark_season.py ---')
    print(format_python_literal(rosters))


if __name__ == '__main__':
    main()
