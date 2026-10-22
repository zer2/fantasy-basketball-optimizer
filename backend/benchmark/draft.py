# benchmark/draft.py
"""Snake-draft engine. Snake logic mirrors move_forward_one_pick (no third-round reversal),
reimplemented here so the UI module is never imported."""

def snake_seat_order(n_drafters, n_starters):
    order = []
    for rnd in range(n_starters):
        seats = range(n_drafters) if rnd % 2 == 0 else reversed(range(n_drafters))
        order.extend(seats)
    return order

def run_draft(seat_agents, n_drafters, n_starters, rng):
    """Run a snake draft.

    ``rng`` is either a single ``np.random.Generator`` shared by every seat
    (original behavior), OR a callable ``rng(seat) -> Generator`` factory. When a
    factory is passed, each seat gets its own persistent generator so that a hero
    swapped at one seat cannot desynchronize the other seats' draws — the common
    random numbers guarantee run_matched_draft relies on.
    """
    player_assignments = {s: [] for s in range(n_drafters)}
    seat_rngs = {s: rng(s) for s in range(n_drafters)} if callable(rng) else None
    for seat in snake_seat_order(n_drafters, n_starters):
        seat_rng = seat_rngs[seat] if seat_rngs is not None else rng
        pick = seat_agents[seat].make_pick(player_assignments, seat, seat_rng)
        if pick is None:
            raise RuntimeError(f'Seat {seat} could not find an eligible pick')
        player_assignments[seat].append(pick)
    return player_assignments
