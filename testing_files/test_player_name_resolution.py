# testing_files/test_player_name_resolution.py
# Name -> NBA id resolution for name-keyed projection sources (PLAYER_NAME_RESOLVER_VIEW, live Snowflake).

import pandas as pd

from backend.data_retrieval import attach_player_ids_by_name, get_player_name_resolvers, strip_accents

_DIABATE_ID = 1631217
_JOKIC_ID = 203999


def test_an_accented_spelling_resolves_when_the_table_has_it_without_accents():
    # HTB writes 'Moussa Diabaté'; the table holds 'Moussa Diabate'. Unresolved, he became a synthetic player, so a
    # live platform's pick of him (matched by the platform's id to his NBA id) never took him off the board.
    frame = attach_player_ids_by_name(pd.DataFrame({'Player': ['Moussa Diabaté', 'Nikola Jokić', 'Nobody Real']}))
    assert frame['player_id'].tolist()[:2] == [_DIABATE_ID, _JOKIC_ID]
    assert pd.isna(frame['player_id'].iloc[2]), 'a name nobody has must still resolve to nothing'


def test_the_accent_free_fallback_never_overrides_an_exact_spelling():
    exact_resolver, accent_free_resolver = get_player_name_resolvers()
    names = list(exact_resolver)
    resolved = attach_player_ids_by_name(pd.DataFrame({'Player': names}))['player_id'].tolist()
    assert resolved == [exact_resolver[name] for name in names]
    assert strip_accents('Pacôme Dadiet') == 'Pacome Dadiet'
    assert accent_free_resolver['Moussa Diabate'] == _DIABATE_ID
