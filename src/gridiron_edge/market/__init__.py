"""Market intelligence utilities for Gridiron Edge.

Pure-math helpers for odds conversion, no-vig debiasing, and Kelly staking.
No data dependencies - every function is a leaf.
"""

from .bet_reference_matching import BetReferenceMatch as BetReferenceMatch
from .bet_reference_matching import BetReferenceMatchStatus as BetReferenceMatchStatus
from .bet_reference_matching import match_bet_references as match_bet_references
from .clv import closing_line_value as closing_line_value
from .clv import spread_clv as spread_clv
from .clv import summarize_clv as summarize_clv
from .clv import total_clv as total_clv
from .edge import classify_edge_strength as classify_edge_strength
from .edge import expected_value as expected_value
from .edge import moneyline_edge as moneyline_edge
from .edge import spread_cover_prob as spread_cover_prob
from .edge import spread_edge as spread_edge
from .edge import total_cover_prob as total_cover_prob
from .edge import total_edge as total_edge
from .history_boundaries import QuoteBoundaryStatus as QuoteBoundaryStatus
from .history_boundaries import QuoteHistoryBoundary as QuoteHistoryBoundary
from .history_boundaries import SelectedQuoteObservation as SelectedQuoteObservation
from .history_boundaries import select_quote_history_boundaries as select_quote_history_boundaries
from .kelly import kelly_fraction as kelly_fraction
from .kelly import kelly_stake as kelly_stake
from .odds_math import NoVigMethod as NoVigMethod
from .odds_math import american_to_decimal as american_to_decimal
from .odds_math import american_to_implied_prob as american_to_implied_prob
from .odds_math import decimal_to_american as decimal_to_american
from .odds_math import hold_pct as hold_pct
from .odds_math import no_vig as no_vig
from .recommendations import build_edge_report as build_edge_report
from .recommendations import compute_game_edges as compute_game_edges
from .recommendations import join_predictions_to_odds as join_predictions_to_odds
from .recommendations import pivot_odds_to_wide as pivot_odds_to_wide
from .recommendations import rank_edges as rank_edges
