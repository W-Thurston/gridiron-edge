import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../../api/schema";
import { useEdges } from "../../api/hooks";
import { TestWrapper } from "../../test/testWrapper";
import { EdgesTable } from "./EdgesTable";

vi.mock("../../api/hooks", () => ({ useEdges: vi.fn() }));
vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

type EdgeList = components["schemas"]["EdgeList"];
type EdgeDiagnostics = components["schemas"]["EdgeDiagnosticsResponse"];
type EdgeBlocker = components["schemas"]["EdgeDiagnosticBlocker"];

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

function response(overrides: Partial<EdgeList> = {}): EdgeList {
  return {
    diagnostics: diagnostics(),
    items: [],
    total: 0,
    bankroll: null,
    kelly_multiplier: 0.25,
    ...overrides,
  };
}

function moneylineEdge(): components["schemas"]["EdgeRow"] {
  return {
    game_id: "2026_01_KC_LAC",
    away_team: "Kansas City Chiefs",
    home_team: "Los Angeles Chargers",
    model_key: "win_prob_elo",
    market_type: "moneyline",
    side: "away",
    model_value: 0.58,
    market_value: 0.45,
    american_odds: 170,
    ev: 0.08,
    edge_strength: "strong",
    kelly_frac: 0.08,
    kelly_stake: 20,
  };
}

function mockLoaded(data: EdgeList) {
  vi.mocked(useEdges).mockReturnValue({
    data,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

const blockers: EdgeBlocker[] = [
  "no_predictions",
  "no_market_data",
  "market_wrong_scope",
  "market_stale",
  "zero_matched_games",
  "incomplete_markets",
];

describe("EdgesTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoaded(response());
  });

  it.each(blockers)("renders blocker %s from diagnostics", (blocker) => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "blocked",
        blockers: [blocker],
        calculated_edge_count: 0,
        positive_edge_count: 0,
        filtered_edge_count: 0,
      }),
    }));

    render(
      <TestWrapper>
        <EdgesTable bankroll={null} kellyMultiplier={0.25} />
      </TestWrapper>,
    );

    expect(screen.getByRole("status")).toHaveAttribute(
      "data-edge-result-kind",
      "blocked",
    );
  });

  it("distinguishes no positive edges", () => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "no_positive_edges",
        positive_edge_count: 0,
        filtered_edge_count: 0,
      }),
    }));

    render(
      <TestWrapper>
        <EdgesTable bankroll={null} kellyMultiplier={0.25} />
      </TestWrapper>,
    );

    expect(screen.getByText("No positive edges.")).toBeInTheDocument();
  });

  it("distinguishes positive edges removed by minimum EV", () => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "positive_edges",
        positive_edge_count: 2,
        filtered_edge_count: 0,
      }),
    }));

    render(
      <TestWrapper>
        <EdgesTable bankroll={null} kellyMultiplier={0.25} />
      </TestWrapper>,
    );

    expect(screen.getByText("No edges passed this filter.")).toBeInTheDocument();
  });

  it("renders real market context and American odds", () => {
    mockLoaded(response({ items: [moneylineEdge()], total: 1 }));

    render(
      <TestWrapper>
        <EdgesTable bankroll={2500} kellyMultiplier={0.25} />
      </TestWrapper>,
    );

    for (const heading of [
      "Matchup",
      "Market",
      "Side",
      "Market Context",
      "Odds",
      "EV",
      "Strength",
    ]) {
      expect(
        screen.getByRole("columnheader", { name: heading }),
      ).toBeInTheDocument();
    }
    expect(screen.getByText("45.0% no-vig")).toBeInTheDocument();
    expect(screen.getByText("+170")).toBeInTheDocument();
  });

  it("labels the Add action with the persisted wager identity", () => {
    mockLoaded(response({ items: [moneylineEdge()], total: 1 }));

    render(
      <TestWrapper>
        <EdgesTable bankroll={2500} kellyMultiplier={0.25} />
      </TestWrapper>,
    );

    expect(
      screen.getByRole("button", {
        name:
          "Add moneyline away for Kansas City Chiefs at Los Angeles Chargers to the Bet Slip",
      }),
    ).toBeInTheDocument();
  });

  it("passes an explicit bankroll and multiplier", () => {
    render(
      <TestWrapper>
        <EdgesTable bankroll={2500} kellyMultiplier={0.1} />
      </TestWrapper>,
    );

    expect(useEdges).toHaveBeenCalledWith({
      bankroll: 2500,
      kelly_multiplier: 0.1,
    });
  });

  it("omits bankroll when unavailable and preserves zero", () => {
    const { rerender } = render(
      <TestWrapper>
        <EdgesTable bankroll={null} kellyMultiplier={0.25} />
      </TestWrapper>,
    );
    expect(useEdges).toHaveBeenLastCalledWith({ kelly_multiplier: 0.25 });

    rerender(
      <TestWrapper>
        <EdgesTable bankroll={0} kellyMultiplier={0.25} />
      </TestWrapper>,
    );
    expect(useEdges).toHaveBeenLastCalledWith({
      bankroll: 0,
      kelly_multiplier: 0.25,
    });
  });
});
