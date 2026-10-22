// ─── Sport config (from GET /config/{sport}) ─────────────────────────────────

export interface ParamOption {
    default: number
    min: number
    max: number | null
}

export interface SportConfig {
    default_categories: string[]
    all_categories: string[]
    short_category_names: Record<string, string>
    options: Record<string, ParamOption>
    positions: Record<string, { base: Record<string, number>; flex: Record<string, number> }>
    position_structure: {
        base_list: string[]
        flex_list: string[]
    }
    position_names: Record<string, string>
}

// ─── Player G-scores (from session creation, pipeline step 4) ────────────────

export interface PlayerGScore {
    player_id: number;
    total: number;
    values: number[];   // per-category G-scores, same order as categories
}

// ─── G-score expectations table ───────────────────────────────────────────────

export interface GScoreRow {
    label: string;
    values: number[];   // per-category score differences; length == categories.length
    total: number;      // sum across categories
    isTotal: boolean;   // true for the summary row (brighter styling)
    isCandidate: boolean;   // the candidate's own row; the others are team-level sums and differentials
}

// ─── Flex position allocations table ──────────────────────────────────────────

export interface FlexRow {
    label: string;
    values: (number | null)[];   // expected fill counts per base position; null = ineligible slot
    isTotal: boolean;
}

export interface FlexAllocations {
    base_positions: string[];   // ordered base position names, e.g. ["PG", "SG", "SF", "PF", "C"]
    rows: FlexRow[];            // one row per flex slot type, plus a total row
}

// ─── Roster grid ──────────────────────────────────────────────────────────────

export interface RosterAssignment {
    player_id: number;      // display resolution (name, headshot) via the player registry
    // candidate = the player being evaluated; hypothetical = a future pick of a hypothetical team
    kind: 'rostered' | 'candidate' | 'hypothetical';
}

export interface Roster {
    slots: string[];                                        // all slot IDs, e.g. ["PG1", "G1", ...]
    assignments: Record<string, RosterAssignment | null>;  // null = empty slot
}

// ─── Hypothetical team ────────────────────────────────────────────────────────

export interface HypotheticalPick {
    player_id: number;
    pick_round: number | null;        // draft: the round the pick is planned for
    predicted_price: number | null;   // auction: whole generic dollars
}

export interface HypotheticalTeam {
    roster: Roster;
    hypothetical_picks: HypotheticalPick[];
}

/** A board as an evaluate request states it: the hypothetical team asks about the board the last evaluate scored. */
export interface BoardRequest {
    player_assignments: Record<string, number[]>;
    my_team_id: string;
    exclusion_list?: number[];
    remaining_cash?: Record<string, number>;
    forced_category_weights?: Record<string, number>;
}

// ─── API request types ────────────────────────────────────────────────────────

export interface ModelSettings {
    // Force-weighting: lets some category weights be fixed by hand, constraining the algorithm into
    // a chosen build. The pinned values themselves ride on the evaluate request, not here — they
    // change the served board but nothing the pipeline builds.
    allow_force_weighting: boolean
    // Window of the truncated-max future-pick model: how many surviving players each
    // future pick effectively chooses among (the punt-aggressiveness dial).
    pick_pool_size: number
    beth: number
    upsilon: number
    psi: number
    chi: number
    aleph: number
    // Peak L1 pulls toward neutral. lambda_c pulls category weights toward balanced;
    // lambda_p pulls flex-position shares toward uniform (its own scale — shares live on
    // a coarser simplex). Named to match the wire fields, since lambda is a Python
    // keyword server-side.
    lambda_c: number
    lambda_p: number
    // How strongly other drafters are modelled as pursuing the punts H-scoring predicts for them,
    // from 0 (neutral pickers with no strategy) to 1 (the prediction taken at face value).
    opponent_model_confidence: number
    n_iterations: number
    streaming_noise: number   // S_σ: SAVOR noise parameter, only meaningful in Auction Mode
    replacement_threshold: number   // S_R: SAVOR replacement threshold in dollars, only meaningful in Auction Mode
    cash_share_softness: number   // s: how early unusable cash starts raising values, only meaningful in Auction Mode
}

export interface DataSource {
    type: 'projections' | 'historical' | 'csv'
    // Keys: 'ESPN', 'DARKO', plus one entry per uploaded custom projection, keyed by its
    // upload data_id (the id is the source's identity; display titles are presentation-only).
    blend_weights: Record<string, number>
    custom_data_ids: string[]
    season?: string | null   // for 'historical' type
}

export interface SessionRequest {
    league: {
        sport: string
        n_drafters: number
        n_picks: number
        scoring_format: string
        // How much of the Head-to-Head objective is winning the majority of categories (0 = each
        // category on its own, 1 = the majority alone). Null under Rotisserie, which uses neither.
        most_categories_weight: number | null
        // The category that counts twice to settle a level matchup; null when none applies.
        tiebreaker_category: string | null
        scoring_system?: 'categories' | 'points'
        categories: string[]
        cash_per_team?: number   // only consulted by the backend in Auction Mode
    }
    // League type: auction sessions require remaining_cash on every evaluate, non-auction
    // sessions forbid it. Patched whenever the user switches modes.
    is_auction: boolean
    // Season Mode builds the base H-scores in one neutral pass instead of the opponent self-play.
    is_season_mode: boolean
    platform: string
    platform_config?: { league_id: string; division_id?: string | null }   // live platforms only
    slot_counts: Record<string, number>
    model_settings: ModelSettings
    data_source: DataSource
    injured_players: number[]
    my_team_id?: string   // provided by seat selector in main content, not sidebar
}

// ─── Top-level player ─────────────────────────────────────────────────────────

export interface AuctionValues {
    your_dollar:   number;   // SAVOR on H-scores, team-specific, current cash/picks, plus unusable_cash_share
    unusable_cash_share: number;   // the part of your_dollar that is the team's otherwise unusable cash
    // The team's spendable cash over what its best options cost, the ratio the smooth cash-share rule acts on; the
    // same for every candidate. null when no available player fits an open slot.
    cash_to_cost_ratio: number | null;
    generic_dollar:   number;   // SAVOR on H-scores, current cash/picks (generic baseline)
    original_dollar:   number;   // SAVOR on H-scores, full original cash/picks
    generic_dollar_g_score: number;   // SAVOR on G-scores, current cash/picks
    original_dollar_g_score: number;   // SAVOR on G-scores, full original cash/picks
}

export interface PlayerResult {
    player_id: number;          // display resolution (name, positions, headshot) via the player registry
    h_score: number;            // overall H-score win rate (0–100 scale)
    h_rank: number;             // rank by H-score among available players
    g_rank: number;             // rank by G-score
    win_rates: number[];        // per-category win rates (0–100, 50 = average)
    category_weights: number[]; // algorithm's relative weighting for future picks (100 = baseline)
    g_score_rows: GScoreRow[];
    flex_allocations?: FlexAllocations;  // absent when position data unavailable
    roster?: Roster;                     // absent when position data unavailable
    auction_values?: AuctionValues;      // present when mode === 'Auction Mode'
    // Points mode candidate properties
    points_per_game?: number;
    total_projected_points?: number;
    marginal_fp?: number;
    delta_fp?: number;
    candidate_games?: number;
    dropped_games?: number;
    daily_deltas?: Record<string, number>;
}

// ─── Points League Types ──────────────────────────────────────────────────────

export interface LeagueTradeImpact {
    pre_trade_points: Record<string, number>
    post_trade_points: Record<string, number>
    pre_trade_ranks: Record<string, number>
    post_trade_ranks: Record<string, number>
    rank_shifts: Record<string, number>
    pre_trade_std: number
    post_trade_std: number
    pre_trade_spread: number
    post_trade_spread: number
    competitive_balance_delta: number
}

