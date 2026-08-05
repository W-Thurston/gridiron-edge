import type { components } from "../../api/schema";

type EdgeDiagnostics =
  components["schemas"]["EdgeDiagnosticsResponse"];
type EdgeBlocker =
  components["schemas"]["EdgeDiagnosticBlocker"];

export type EdgeResultPresentation = {
  kind: "blocked" | "empty" | "filtered";
  title: string;
  detail: string;
  blockerMessages: string[];
};

const BLOCKER_MESSAGES: Record<EdgeBlocker, string> = {
  no_predictions: "Weekly predictions are unavailable.",
  no_market_data: "Market data is unavailable for this week.",
  market_wrong_scope:
    "Available market data belongs to a different season or week.",
  market_stale: "Market data is stale.",
  zero_matched_games:
    "Predictions and markets did not match any scheduled games.",
  incomplete_markets:
    "One or more games have incomplete market coverage.",
};

export function getEdgeResultPresentation(
  diagnostics: EdgeDiagnostics,
): EdgeResultPresentation | null {
  if (diagnostics.state === "blocked") {
    const blockerMessages = diagnostics.blockers.map(
      (blocker) => BLOCKER_MESSAGES[blocker],
    );
    return {
      kind: "blocked",
      title: "Weekly edges are unavailable.",
      detail:
        blockerMessages.length === 0
          ? "The edge service did not provide a blocker reason."
          : blockerMessages.join(" "),
      blockerMessages,
    };
  }

  if (diagnostics.state === "no_calculable_edges") {
    return {
      kind: "empty",
      title: "No calculable edges.",
      detail: "No markets had enough information to calculate an edge.",
      blockerMessages: [],
    };
  }

  if (diagnostics.state === "no_positive_edges") {
    return {
      kind: "empty",
      title: "No positive edges.",
      detail: "Markets were evaluated, but no positive edges were found.",
      blockerMessages: [],
    };
  }

  if (
    diagnostics.state === "positive_edges" &&
    diagnostics.positive_edge_count > 0 &&
    diagnostics.filtered_edge_count === 0
  ) {
    return {
      kind: "filtered",
      title: "No edges passed this filter.",
      detail:
        "Positive edges exist, but none passed the requested minimum EV.",
      blockerMessages: [],
    };
  }

  return null;
}
