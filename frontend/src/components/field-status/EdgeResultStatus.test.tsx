import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { EdgeResultStatus } from "./EdgeResultStatus";

type EdgeDiagnostics =
  components["schemas"]["EdgeDiagnosticsResponse"];

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

describe("EdgeResultStatus", () => {
  it("renders the authoritative blocked state", () => {
    render(
      <EdgeResultStatus
        diagnostics={diagnostics({
          state: "blocked",
          blockers: ["no_market_data"],
        })}
      />,
    );

    expect(screen.getByRole("status")).toHaveAttribute(
      "data-edge-result-kind",
      "blocked",
    );
    expect(
      screen.getByText("Weekly edges are unavailable."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Market data is unavailable for this week."),
    ).toBeInTheDocument();
  });

  it("lists every simultaneous blocker", () => {
    render(
      <EdgeResultStatus
        diagnostics={diagnostics({
          state: "blocked",
          blockers: ["no_predictions", "incomplete_markets"],
        })}
      />,
    );

    const blockerItems = screen.getAllByRole("listitem");
    expect(blockerItems).toHaveLength(2);
    expect(blockerItems[0]).toHaveTextContent(
      "Weekly predictions are unavailable.",
    );
    expect(blockerItems[1]).toHaveTextContent(
      "One or more games have incomplete market coverage.",
    );
  });

  it("renders filtered positive edges separately", () => {
    render(
      <EdgeResultStatus
        compact
        diagnostics={diagnostics({
          state: "positive_edges",
          positive_edge_count: 2,
          filtered_edge_count: 0,
        })}
      />,
    );

    expect(screen.getByRole("status")).toHaveAttribute(
      "data-edge-result-kind",
      "filtered",
    );
    expect(
      screen.getByText("No edges passed this filter."),
    ).toBeInTheDocument();
  });

  it("renders nothing when positive edges are returned", () => {
    const { container } = render(
      <EdgeResultStatus diagnostics={diagnostics()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
