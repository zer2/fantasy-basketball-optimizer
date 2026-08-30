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
    isCandidate: boolean;   // true = the player being evaluated; false = already rostered
}

export interface Roster {
    slots: string[];                                        // all slot IDs, e.g. ["PG1", "G1", ...]
    assignments: Record<string, RosterAssignment | null>;  // null = empty slot
}

// ─── API request types ────────────────────────────────────────────────────────

export interface ModelParameters {
    omega: number
    gamma: number
    beth: number
    upsilon: number
    psi: number
    chi: number
    aleph: number
    kappa: number
    // Peak L1 pull of category weights toward neutral (the algorithm's lambda). Named
    // reg_lambda to match the wire field, since lambda is a Python keyword server-side.
    reg_lambda: number
    // How strongly other drafters are modelled as pursuing the punts H-scoring predicts for them,
    // from 0 (neutral pickers with no strategy) to 1 (the prediction taken at face value).
    opponent_model_confidence: number
    n_iterations: number
    streaming_noise: number   // S_σ: SAVOR noise parameter, only meaningful in Auction Mode
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
        categories: string[]
        cash_per_team?: number   // only consulted by the backend in Auction Mode
    }
    // League type: auction sessions require remaining_cash on every evaluate, non-auction
    // sessions forbid it. Patched whenever the user switches modes.
    is_auction: boolean
    platform: string
    platform_config?: { league_id: string; division_id?: string | null }   // live platforms only
    slot_counts: Record<string, number>
    parameters: ModelParameters
    data_source: DataSource
    injured_players: string[]
    my_team_id?: string   // provided by seat selector in main content, not sidebar
}

// ─── Top-level player ─────────────────────────────────────────────────────────

export interface AuctionValues {
    your_dollar:   number;   // SAVOR on H-scores, team-specific, current cash/picks
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
}
