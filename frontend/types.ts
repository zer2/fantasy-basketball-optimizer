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
    values: number[];   // expected fill counts per base position; -999 = ineligible slot
    isTotal: boolean;
}

export interface FlexAllocations {
    base_positions: string[];   // ordered base position names, e.g. ["PG", "SG", "SF", "PF", "C"]
    rows: FlexRow[];            // one row per flex slot type, plus a total row
}

// ─── Roster grid ──────────────────────────────────────────────────────────────

export interface RosterAssignment {
    name: string;
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
    n_iterations: number
}

export interface DataSource {
    type: 'projections' | 'historical'
    blend_weights: { espn: number; darko: number; htb: number; bbm: number }
    custom_data_ids: { HTB: string | null; BBM: string | null }
}

export interface SessionRequest {
    league: {
        sport: string
        n_drafters: number
        n_picks: number
        scoring_format: string
        categories: string[]
    }
    platform: string
    slot_counts: Record<string, number>
    parameters: ModelParameters
    data_source: DataSource
    injured_players: string[]
    my_team_id: string
}

// ─── Top-level player ─────────────────────────────────────────────────────────

export interface Player {
    name: string;
    h_score: number;            // overall H-score win rate (0–100 scale)
    h_rank: number;             // rank by H-score among available players
    g_rank: number;             // rank by G-score
    win_rates: number[];        // per-category win rates (0–100, 50 = average)
    category_weights: number[]; // algorithm's relative weighting for future picks (100 = baseline)
    g_score_rows: GScoreRow[];
    flex_allocations: FlexAllocations;
    roster: Roster;
}
