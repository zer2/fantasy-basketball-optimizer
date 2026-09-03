"""Pre-computed position structure, derived once from sport sport_params + slot_counts.

WHY IT EXISTS: the Streamlit original read the position structure out of
st.session_state wherever it was needed. This is the explicit-argument replacement —
derive the structure ONCE and pass the resulting PositionConfig down to the math
functions, so no two callers can silently drift apart. (Not recomputing the same
thing in every helper is a side benefit, not the point.)

WHAT IT IS: four views of the same position structure, each convenient in a
different context — see the field comments below.

It lives under math/ because math/ is its only consumer (position_optimization and
algorithm_agents); nothing about it is session state.
"""

from dataclasses import dataclass


@dataclass
class PositionConfig:
    # The raw sport-level structure: {'base_list': [...], 'flex_list': [...], 'flex': {...}}.
    position_structure: dict

    # Roster slots per position, e.g. {'PG': 1, 'SG': 1, 'G': 1, 'C': 2, ...}.
    position_numbers:   dict[str, int]

    # Slot-layout map. A roster is a flat vector of slots laid out contiguously by
    # position, so position_ranges[pos] = {'start', 'end'} is the [start, end) slice of
    # that vector owned by `pos`. Lets the math address a position's slots directly
    # instead of rebuilding the layout at each call site.
    position_ranges:    dict[str, dict]

    # For each FLEX position, the indices INTO base_list of the base positions eligible
    # to fill it — e.g. 'G' -> [0, 1] when base_list starts [PG, SG]. Used to decide
    # which players may legally occupy a flex slot.
    position_indices:   dict[str, list[int]]


def build_position_config(
    sport_params: dict
    , slot_counts: dict
) -> PositionConfig:
    """Build a PositionConfig from sport-level sport_params and the session's slot_counts."""
    position_structure = sport_params['position_structure']
    base_list  = position_structure['base_list']
    flex_list  = position_structure['flex_list']
    all_positions = base_list + flex_list

    position_numbers = {pos: slot_counts.get(pos, 0) for pos in all_positions}

    # Hand each position, in order, a contiguous block of slot indices — this is what
    # makes the roster vector's layout predictable to every downstream consumer.
    start = 0
    position_ranges: dict[str, dict] = {}
    for pos in all_positions:
        end = start + position_numbers[pos]
        position_ranges[pos] = {'start': start, 'end': end}
        start = end

    flex_info = position_structure['flex']
    position_indices = {
        pos: [i for i, val in enumerate(base_list) if val in flex_info[pos]['bases']]
        for pos in flex_list
    }

    return PositionConfig(
        position_structure = position_structure,
        position_numbers   = position_numbers,
        position_ranges    = position_ranges,
        position_indices   = position_indices,
    )
