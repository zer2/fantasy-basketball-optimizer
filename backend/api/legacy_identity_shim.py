"""TEMPORARY (Stage A of the player-identity refactor; deleted with the Stage-B contract).

The services and math now speak player ids, but the HTTP contract still speaks the
string-identity era's 'Name (POS)' labels. These helpers convert at the router boundary
in both directions so the wire bytes stay identical while the interior re-keys:
  - inbound: legacy labels -> ids (raising an actionable 400 for unknown labels, the
    same failure surface UnknownRosterPlayersError provides at the service tier);
  - outbound: ids -> legacy labels via build_legacy_display_label.
"""

from __future__ import annotations

from fastapi import HTTPException

from backend.player_identity import build_legacy_display_label


def build_label_to_player_id_map(player_registry: dict) -> dict[str, int]:
    """{legacy 'Name (POS)' label -> player id} over a session's registry."""
    return {
        build_legacy_display_label(identity): identity.player_id
        for identity in player_registry.values()
    }


def convert_labels_to_player_ids(
    player_registry: dict
    , labels: list[str]
) -> list[int]:
    """Inbound edge: legacy labels -> player ids, 400 on any label the registry lacks."""
    label_map = build_label_to_player_id_map(player_registry)
    unknown = [label for label in labels if label not in label_map]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail='These rostered players are not in the current player pool: '
                   + ', '.join(sorted(unknown))
                   + '. The data-source change altered the pool; clear the board or '
                     'restore the previous sources.',
        )
    return [label_map[label] for label in labels]


def convert_assignments_to_player_ids(
    player_registry: dict
    , player_assignments: dict[str, list[str]]
) -> dict[str, list[int]]:
    """Inbound edge for whole boards: {team -> [label]} -> {team -> [id]}."""
    label_map = build_label_to_player_id_map(player_registry)
    unknown = sorted({
        label for labels in player_assignments.values() for label in labels
        if label not in label_map
    })
    if unknown:
        raise HTTPException(
            status_code=400,
            detail='These rostered players are not in the current player pool: '
                   + ', '.join(unknown)
                   + '. The data-source change altered the pool; clear the board or '
                     'restore the previous sources.',
        )
    return {
        team: [label_map[label] for label in labels]
        for team, labels in player_assignments.items()
    }


def convert_player_ids_to_labels(
    player_registry: dict
    , player_ids: list[int]
) -> list[str]:
    """Outbound edge: ids -> legacy labels."""
    return [build_legacy_display_label(player_registry[player_id]) for player_id in player_ids]
