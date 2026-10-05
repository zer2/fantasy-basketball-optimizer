-- PLAYER_NAME_RESOLVER_VIEW: every known spelling of every player -> one NBA_PLAYER_ID.
--
-- Projections reach the app keyed by name, and from anywhere: ESPN, DARKO, Hashtag Basketball, Basketball Monster,
-- a hand-edited file, or a blend of several. Each source spells players its own way, so a name is resolved against
-- every spelling UNIFIED_PLAYER_TABLE knows (all of its name columns), not just the canonical one.
--
-- A spelling that belongs to more than one player (11 of 2,885 when written: Gary Trent, Glen Rice, Patrick Ewing and
-- other father/son or same-name pairs) is decided by career span:
--   1. the most recently active player wins: the latest LAST_SEASON (the season a career's last year starts in;
--      every currently active player carries the current one);
--   2. between players equally recent -- in practice, two active players -- the latest FIRST_SEASON wins, so a
--      rookie takes the name over an established player;
--   3. then the higher, newer NBA id.
-- Career spans come from NBA_PLAYER_ATTRIBUTE_TABLE. An UNDEBUTED rookie has none there (the 2026 draft class was
-- absent when this was written) and no box score either, so a player with neither is taken to be one: both seasons
-- are set to the coming season (the latest LAST_SEASON on record), which ranks him first. Box scores reach back to
-- 1984-85, so the only others this could catch are pre-1984 players missing from both tables.
-- Any other player with no career span on record ranks after every player with one.
--
-- CANDIDATE_COUNT keeps collisions visible: > 1 means the spelling was contested and this row won.
-- Rows without an NBA_PLAYER_ID are left out; a name only they carry resolves to nothing, like an unknown name.

CREATE OR REPLACE VIEW PLAYER_NAME_RESOLVER_VIEW AS
WITH SPELLINGS AS (
    -- UNPIVOT drops NULL names; DISTINCT collapses a player spelled the same way in several columns.
    SELECT DISTINCT PLAYER_NAME
                  , NBA_PLAYER_ID
    FROM UNIFIED_PLAYER_TABLE
    UNPIVOT (PLAYER_NAME FOR NAME_COLUMN IN (
        MASTER_PLAYER_NAME, DARKO_NAME, ESPN_NAME, ROTOWIRE_NAME, HTB_NAME, BBM_NAME
    ))
    WHERE NBA_PLAYER_ID IS NOT NULL
),
COMING_SEASON AS (
    SELECT MAX(TRY_TO_NUMBER(LAST_SEASON)) AS SEASON
    FROM NBA_PLAYER_ATTRIBUTE_TABLE
),
CAREER_SPANS AS (
    SELECT SPELLINGS.NBA_PLAYER_ID
         , CASE WHEN ATTRIBUTES.LAST_SEASON IS NULL AND BOX_SCORED.NBA_PLAYER_ID IS NULL
                THEN COMING_SEASON.SEASON
                ELSE TRY_TO_NUMBER(ATTRIBUTES.FIRST_SEASON) END AS FIRST_SEASON
         , CASE WHEN ATTRIBUTES.LAST_SEASON IS NULL AND BOX_SCORED.NBA_PLAYER_ID IS NULL
                THEN COMING_SEASON.SEASON
                ELSE TRY_TO_NUMBER(ATTRIBUTES.LAST_SEASON) END AS LAST_SEASON
    FROM (SELECT DISTINCT NBA_PLAYER_ID FROM SPELLINGS) AS SPELLINGS
    CROSS JOIN COMING_SEASON
    LEFT JOIN NBA_PLAYER_ATTRIBUTE_TABLE AS ATTRIBUTES
      ON ATTRIBUTES.NBA_PLAYER_ID = SPELLINGS.NBA_PLAYER_ID
    LEFT JOIN (SELECT DISTINCT NBA_PLAYER_ID FROM BOX_SCORE_TABLE) AS BOX_SCORED
      ON BOX_SCORED.NBA_PLAYER_ID = SPELLINGS.NBA_PLAYER_ID
)
SELECT SPELLINGS.PLAYER_NAME
     , SPELLINGS.NBA_PLAYER_ID
     , CAREER_SPANS.FIRST_SEASON
     , CAREER_SPANS.LAST_SEASON
     , COUNT(*) OVER (PARTITION BY SPELLINGS.PLAYER_NAME) AS CANDIDATE_COUNT
FROM SPELLINGS
JOIN CAREER_SPANS
  ON CAREER_SPANS.NBA_PLAYER_ID = SPELLINGS.NBA_PLAYER_ID
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY SPELLINGS.PLAYER_NAME
    ORDER BY CAREER_SPANS.LAST_SEASON  DESC NULLS LAST
           , CAREER_SPANS.FIRST_SEASON DESC NULLS LAST
           , SPELLINGS.NBA_PLAYER_ID DESC
) = 1;
