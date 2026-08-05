import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { useEdges, useGame, usePropsList } from "../api/hooks";
import { TestWrapper } from "../test/testWrapper";
import { GameDetail } from "./GameDetail";

vi.mock("../api/hooks", () => ({
  useEdges: vi.fn(),
  useGame: vi.fn(),
  usePropsList: vi.fn(),
}));
vi.mock("../context/NavContext", () => ({
  useNav: vi.fn(() => ({
    route: { path: "/games", params: { gameId: "2026_01_KC_LAC" } },
    navigate: vi.fn(),
  })),
  NavProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

type Detail = components["schemas"]["GameDetail"];
type EdgeList = components["schemas"]["EdgeList"];

function detail(overrides: Partial<Detail> = {}): Detail {
  return {
    game_id: "2026_01_KC_LAC",
    game_date: "2026-09-05",
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
    ...overrides,
  };
}

function edges(): EdgeList {
  return {
    diagnostics: {
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
      positive_edge_count: 0,
      filtered_edge_count: 0,
      state: "no_positive_edges",
      blockers: [],
    },
    items: [],
    total: 0,
  };
}

function mockLoaded(data: Detail) {
  vi.mocked(useGame).mockReturnValue({
    data,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useEdges).mockReturnValue({
    data: edges(),
    isLoading: false,
    error: null,
  } as never);
  vi.mocked(usePropsList).mockReturnValue({
    data: { items: [], total: 0 },
    isLoading: false,
    error: null,
  } as never);
}

describe("GameDetail weekly component readiness", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoaded(detail());
  });

  it("shows missing Win while retaining an independent projected score", () => {
    mockLoaded(detail({ win: { status: "forecast_missing" } }));
    render(<TestWrapper><GameDetail /></TestWrapper>);
    expect(screen.getAllByText("Win forecast missing").length).toBeGreaterThan(0);
    expect(screen.getByText("Kansas City Chiefs 22.5")).toBeInTheDocument();
    expect(screen.getByText("Los Angeles Chargers 25.0")).toBeInTheDocument();
  });

  it("shows Spread calibration unavailability independently", () => {
    mockLoaded(detail({
      spread: { status: "calibration_unavailable" },
      projected_score: { status: "spread_unavailable" },
    }));
    render(<TestWrapper><GameDetail /></TestWrapper>);
    expect(screen.getAllByText("Spread calibration unavailable").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Projected score unavailable because Spread is unavailable"),
    ).toBeInTheDocument();
  });

  it("keeps the Total point estimate when uncertainty is unavailable", () => {
    mockLoaded(detail({
      total: { status: "uncertainty_unavailable", model_total: 47.5 },
    }));
    render(<TestWrapper><GameDetail /></TestWrapper>);
    expect(screen.getByText("O 47.5")).toBeInTheDocument();
    expect(screen.getByText("U 47.5")).toBeInTheDocument();
  });

  it("does not render synthetic uncertainty-band copy", () => {
    render(<TestWrapper><GameDetail /></TestWrapper>);
    expect(screen.queryByText(/uncertainty band/i)).not.toBeInTheDocument();
  });
});


describe("GameDetail edge readiness", () => {
  it("shows a weekly blocker instead of No model edge or No play", () => {
    mockLoaded(detail());
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...edges(),
        diagnostics: {
          ...edges().diagnostics,
          state: "blocked",
          blockers: ["no_market_data"],
          calculated_edge_count: 0,
        },
      },
      isLoading: false,
      error: null,
    } as never);

    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(
      screen.getAllByText("Market data is unavailable for this week.").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("No model edge")).not.toBeInTheDocument();
    expect(screen.queryByText("No play")).not.toBeInTheDocument();
  });

  it("allows No play only for a completed no-positive-edge result", () => {
    mockLoaded(detail());
    render(<TestWrapper><GameDetail /></TestWrapper>);
    expect(screen.getAllByText("No play")).toHaveLength(3);
  });

  it("shows filtered positive edges instead of No play", () => {
    mockLoaded(detail());
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...edges(),
        diagnostics: {
          ...edges().diagnostics,
          state: "positive_edges",
          positive_edge_count: 2,
          filtered_edge_count: 0,
        },
      },
      isLoading: false,
      error: null,
    } as never);

    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(
      screen.getAllByText("No edges passed this filter.").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("No play")).not.toBeInTheDocument();
  });

  it("shows a game-specific empty state when positive edges belong elsewhere", () => {
    mockLoaded(detail());
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...edges(),
        diagnostics: {
          ...edges().diagnostics,
          state: "positive_edges",
          positive_edge_count: 1,
          filtered_edge_count: 1,
        },
        items: [{
          game_id: "2026_01_BUF_NYJ",
          away_team: "Buffalo Bills",
          home_team: "New York Jets",
          model_key: "win_prob_elo",
          market_type: "moneyline",
          side: "away",
          model_value: 0.6,
          market_value: 0.52,
          american_odds: -110,
          ev: 0.08,
          edge_strength: "strong",
        }],
      },
      isLoading: false,
      error: null,
    } as never);

    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(
      screen.getByText("No positive edge for this game."),
    ).toBeInTheDocument();
  });

  it("renders persisted market context and American odds", () => {
    mockLoaded(detail());
    const base = edges();
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...base,
        diagnostics: {
          ...base.diagnostics,
          state: "positive_edges",
          positive_edge_count: 3,
          filtered_edge_count: 3,
        },
        items: [
          {
            game_id: "2026_01_KC_LAC",
            away_team: "Kansas City Chiefs",
            home_team: "Los Angeles Chargers",
            model_key: "spread_elo",
            market_type: "spread",
            side: "home",
            model_value: -2.5,
            market_value: -3.5,
            american_odds: -110,
            ev: 0.06,
            edge_strength: "moderate",
          },
          {
            game_id: "2026_01_KC_LAC",
            away_team: "Kansas City Chiefs",
            home_team: "Los Angeles Chargers",
            model_key: "total_total",
            market_type: "total",
            side: "over",
            model_value: 47.5,
            market_value: 46.5,
            american_odds: 105,
            ev: 0.05,
            edge_strength: "moderate",
          },
          {
            game_id: "2026_01_KC_LAC",
            away_team: "Kansas City Chiefs",
            home_team: "Los Angeles Chargers",
            model_key: "win_prob_elo",
            market_type: "moneyline",
            side: "home",
            model_value: 0.58,
            market_value: 0.52,
            american_odds: -120,
            ev: 0.04,
            edge_strength: "lean",
          },
        ],
      },
      isLoading: false,
      error: null,
    } as never);

    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getByText("Home -3.5 · -110")).toBeInTheDocument();
    expect(screen.getByText("Total 46.5 · +105")).toBeInTheDocument();
    expect(screen.getByText("52.0% no-vig · -120")).toBeInTheDocument();
  });
});
