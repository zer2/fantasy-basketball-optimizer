"""The built scoring model for a session: the H-scoring agent plus the data it scores.

Produced by services.build_scorer (the pipeline); consumed by ranking and trading. Because the
agent and its baseline cache live together here, rebuilding the agent (any pipeline run reaches
the final step) yields a Scorer whose generic_h_scores is reset — the cache can't outlive the
agent it belongs to, which used to be an invariant maintained by hand across several files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Scorer:
    # info: processed player data — G-scores, X-scores, positions, covariance, etc. (the output
    #   of process_player_data, and what the HAgent is built from).
    info: dict
    # h_agent: the HAgent — the H-score gradient engine. None only transiently, while the pipeline
    #   is between building `info` (step 4) and (re)building the agent (step 5).
    h_agent: Optional[Any] = None
    # generic_h_scores: cached baseline (empty-board) H-score ranking, populated on the first
    #   ranking/trading call and reused for auction dollar values. pd.Series when set; reset to
    #   None whenever the agent is rebuilt.
    generic_h_scores: Optional[Any] = None
