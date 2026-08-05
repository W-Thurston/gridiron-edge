import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { getEdgeResultPresentation } from "./edgeResultStatus";

type EdgeDiagnostics =
  components["schemas"]["EdgeDiagnosticsResponse"];
type EdgeBlocker =
  components["schemas"]["EdgeDiagnosticBlocker"];

function diagnostics(
  overrides: Partial<EdgeDiagnostics> = {},
): EdgeDiagnostics {
  return {
    season: "2026-2027",
    week: 1,
    prediction_game_count: 1,
    market_game_count: 1,
    matched_game_count: 1,
    complete_moneyline_count: 1,
    complete_spread_count: 1,
    complete_total_count: 1,
    eligible_market_count: 3,
    calculated_edge_count: 3,
    positive_edge_count: 1,
    filtered_edge_count: 1,
    state: "positive_edges",
    blockers: [],
    ...overrides,
  };
}

const blockerCases: Array<[EdgeBlocker, string]> = [
  ["no_predictions", "Weekly predictions are unavailable."],
  ["no_market_data", "Market data is unavailable for this week."],
  [
    "market_wrong_scope",
    "Available market data belongs to a different season or week.",
  ],
  ["market_stale", "Market data is stale."],
  [
    "zero_matched_games",
    "Predictions and markets did not match any scheduled games.",
  ],
  [
    "incomplete_markets",
    "One or more games have incomplete market coverage.",
  ],
];

describe("getEdgeResultPresentation", () => {
  it.each(blockerCases)(
    "maps blocker %s without changing its meaning",
    (blocker, message) => {
      const result = getEdgeResultPresentation(
        diagnostics({ state: "blocked", blockers: [blocker] }),
      );

      expect(result).toEqual({
        kind: "blocked",
        title: "Weekly edges are unavailable.",
        detail: message,
        blockerMessages: [message],
      });
    },
  );

  it("preserves every simultaneous blocker in service order", () => {
    const result = getEdgeResultPresentation(
      diagnostics({
        state: "blocked",
        blockers: ["no_predictions", "no_market_data"],
      }),
    );

    expect(result?.blockerMessages).toEqual([
      "Weekly predictions are unavailable.",
      "Market data is unavailable for this week.",
    ]);
    expect(result?.detail).toBe(
      "Weekly predictions are unavailable. " +
        "Market data is unavailable for this week.",
    );
  });

  it("represents blocked results that lack a blocker reason", () => {
    const result = getEdgeResultPresentation(
      diagnostics({ state: "blocked", blockers: [] }),
    );

    expect(result).toEqual({
      kind: "blocked",
      title: "Weekly edges are unavailable.",
      detail: "The edge service did not provide a blocker reason.",
      blockerMessages: [],
    });
  });

  it("distinguishes no calculable edges", () => {
    expect(
      getEdgeResultPresentation(
        diagnostics({
          state: "no_calculable_edges",
          calculated_edge_count: 0,
          positive_edge_count: 0,
          filtered_edge_count: 0,
        }),
      ),
    ).toMatchObject({
      kind: "empty",
      title: "No calculable edges.",
    });
  });

  it("distinguishes no positive edges", () => {
    expect(
      getEdgeResultPresentation(
        diagnostics({
          state: "no_positive_edges",
          positive_edge_count: 0,
          filtered_edge_count: 0,
        }),
      ),
    ).toMatchObject({
      kind: "empty",
      title: "No positive edges.",
    });
  });

  it("distinguishes positive edges removed by minimum EV", () => {
    expect(
      getEdgeResultPresentation(
        diagnostics({
          state: "positive_edges",
          positive_edge_count: 3,
          filtered_edge_count: 0,
        }),
      ),
    ).toEqual({
      kind: "filtered",
      title: "No edges passed this filter.",
      detail:
        "Positive edges exist, but none passed the requested minimum EV.",
      blockerMessages: [],
    });
  });

  it("returns no empty presentation when positive edges are returned", () => {
    expect(getEdgeResultPresentation(diagnostics())).toBeNull();
  });
});
