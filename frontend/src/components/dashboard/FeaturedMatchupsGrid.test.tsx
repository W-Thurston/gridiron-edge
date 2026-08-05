import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../../api/schema";
import { useEdges, useGamesList } from "../../api/hooks";
import { TestWrapper } from "../../test/testWrapper";
import { FeaturedMatchupsGrid } from "./FeaturedMatchupsGrid";

vi.mock("../../api/hooks", () => ({
  useEdges: vi.fn(),
  useGamesList: vi.fn(),
}));

vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

type EdgeList = components["schemas"]["EdgeList"];
type EdgeDiagnostics = components["schemas"]["EdgeDiagnosticsResponse"];
type GameList = components["schemas"]["GameList"];

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

function edgeResponse(overrides: Partial<EdgeList> = {}): EdgeList {
  return {
    diagnostics: diagnostics(),
    items: [],
    total: 0,
    ...overrides,
  };
}

function gameResponse(): GameList {
  return {
    season: "2026-2027",
    week: 1,
    total: 1,
    items: [
      {
        game_id: "2026_01_KC_LAC",
        game_date: "2026-09-05",
        season: "2026-2027",
        week: 1,
        away_team: "Kansas City Chiefs",
        home_team: "Los Angeles Chargers",
        win: {
          status: "available",
          away_win_prob: 0.42,
          home_win_prob: 0.58,
        },
        spread: { status: "available", model_spread: -2.5 },
        total: { status: "available", model_total: 47.5 },
        projected_score: { status: "available", away: 22.5, home: 25 },
      },
    ],
  };
}

function mockLoaded(edges: EdgeList, games: GameList = gameResponse()) {
  vi.mocked(useEdges).mockReturnValue({
    data: edges,
    isLoading: false,
    error: null,
  } as never);
  vi.mocked(useGamesList).mockReturnValue({
    data: games,
    isLoading: false,
    error: null,
  } as never);
}

describe("FeaturedMatchupsGrid", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useEdges).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);
    vi.mocked(useGamesList).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);
  });

  it("renders loading state", () => {
    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );

    expect(screen.getByText("Featured Matchups")).toBeInTheDocument();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("shows missing predictions without claiming the schedule is empty", () => {
    mockLoaded(
      edgeResponse({
        diagnostics: diagnostics({
          state: "blocked",
          blockers: ["no_predictions"],
          prediction_game_count: 0,
          calculated_edge_count: 0,
          positive_edge_count: 0,
          filtered_edge_count: 0,
        }),
      }),
    );

    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );

    expect(
      screen.getByText("Weekly predictions are unavailable."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/No featured matchups yet/)).not.toBeInTheDocument();
  });

  it("shows no-positive-edge analysis as a valid empty result", () => {
    mockLoaded(
      edgeResponse({
        diagnostics: diagnostics({
          state: "no_positive_edges",
          positive_edge_count: 0,
          filtered_edge_count: 0,
        }),
      }),
    );

    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );

    expect(screen.getByText("No positive edges.")).toBeInTheDocument();
    expect(
      screen.getByText("Markets were evaluated, but no positive edges were found."),
    ).toBeInTheDocument();
  });

  it("renders a positive edge joined to its scheduled game", () => {
    mockLoaded(
      edgeResponse({
        total: 1,
        items: [
          {
            game_id: "2026_01_KC_LAC",
            game_date: "2026-09-05",
            season: "2026-2027",
            week: 1,
            away_team: "Kansas City Chiefs",
            home_team: "Los Angeles Chargers",
            model_key: "win_prob_elo",
            market_type: "moneyline",
            side: "home",
            model_value: 0.58,
            market_value: 0.52,
            american_odds: -110,
            ev: 0.08,
            edge_strength: "strong",
          },
        ],
      }),
    );

    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );

    expect(screen.getByText("Home win prob 58%")).toBeInTheDocument();
    expect(screen.getByText("+8.0% EV")).toBeInTheDocument();
  });
});
