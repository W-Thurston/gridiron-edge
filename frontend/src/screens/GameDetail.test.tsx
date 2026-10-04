import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { useEdges, useGame, usePropsList } from "../api/hooks";
import { TestWrapper } from "../test/testWrapper";
import { GameDetail } from "./GameDetail";
import type { Recommendation } from "../components/recommendations/recommendationPresentation"
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
    starting_qb: { status: "not_recorded" },
    ...overrides,
  };
}

function edges(): EdgeList {
  return {
    freshness: { evaluated_at: "2026-10-04T12:00:00Z", max_age_seconds: 86400 },
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

  it("emphasizes the home team only when its projected score is higher", () => {
    render(<TestWrapper><GameDetail /></TestWrapper>);

    const away = screen.getByText("Kansas City Chiefs 22.5");
    const home = screen.getByText("Los Angeles Chargers 25.0");
    expect(away).toHaveStyle({ color: "var(--ink-2)" });
    expect(home).toHaveStyle({ color: "var(--pos)" });
  });

  it("emphasizes the away team when its projected score is higher", () => {
    mockLoaded(detail({
      spread: { status: "available", model_spread: 3.5 },
      projected_score: { status: "available", away: 27.5, home: 24 },
    }));
    render(<TestWrapper><GameDetail /></TestWrapper>);

    const away = screen.getByText("Kansas City Chiefs 27.5");
    const home = screen.getByText("Los Angeles Chargers 24.0");
    expect(away).toHaveStyle({ color: "var(--pos)" });
    expect(home).toHaveStyle({ color: "var(--ink-2)" });
  });

  it("uses equal neutral emphasis for tied projected scores", () => {
    mockLoaded(detail({
      spread: { status: "available", model_spread: 0 },
      projected_score: { status: "available", away: 24, home: 24 },
    }));
    render(<TestWrapper><GameDetail /></TestWrapper>);

    const away = screen.getByText("Kansas City Chiefs 24.0");
    const home = screen.getByText("Los Angeles Chargers 24.0");
    expect(away).toHaveStyle({ color: "var(--ink-2)" });
    expect(home).toHaveStyle({ color: "var(--ink-2)" });
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


  it("uses the persisted sportsbook subset for the game lean and market recommendation", () => {
    localStorage.setItem("hm-app", JSON.stringify({
      sportsbookMode: "selected",
      selectedSportsbooks: ["draftkings"],
    }));
    mockLoaded(detail());
    const base = edges();
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...base,
        diagnostics: {
          ...base.diagnostics,
          state: "positive_edges",
          positive_edge_count: 2,
          filtered_edge_count: 2,
        },
        items: [
          {
            game_id: "2026_01_KC_LAC",
            away_team: "Kansas City Chiefs",
            home_team: "Los Angeles Chargers",
            model_key: "win_prob_elo",
            market_type: "moneyline",
            side: "home",
            model_value: 0.58,
            market_value: 0.52,
            provider: "the_odds_api",
            provider_event_id: "event-dk",
            sportsbook: "draftkings",
            american_odds: -150,
            ev: 0.08,
            edge_strength: "strong",
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
            provider: "the_odds_api",
            provider_event_id: "event-fd",
            sportsbook: "fanduel",
            american_odds: -140,
            ev: 0.1,
            edge_strength: "strong",
          },
        ],
      },
      isLoading: false,
      error: null,
    } as never);

    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(
      screen.getAllByText((_, element) =>
        element?.textContent === "DraftKings · -150 · +8.0% EV"
      ).length,
    ).toBeGreaterThan(0);
    expect(
      screen.queryByText((_, element) =>
        element?.textContent === "FanDuel · -140 · +10.0% EV"
      ),
    ).not.toBeInTheDocument();
  });

});

describe("GameDetail recommendation presentation and action separation", () => {
  function edgeWithRecommendation(
    recommendation: Recommendation | null,
  ): components["schemas"]["EdgeRow"] {
    return {
      game_id: "2026_01_KC_LAC",
      away_team: "Kansas City Chiefs",
      home_team: "Los Angeles Chargers",
      model_key: "win_prob_elo",
      market_type: "moneyline",
      side: "home",
      model_value: 0.58,
      market_value: 0.52,
      american_odds: -120,
      ev: 0.08,
      edge_strength: "strong",
      is_live: false,
      recommendation,
    };
  }

  function persisted(overrides: Partial<Recommendation>): Recommendation {
    return {
      result_id: "result-1",
      decision_state: "recommendation_eligible",
      result_state: "recommended",
      recommendation_eligible: true,
      evaluated_at: "2026-09-01T12:10:00Z",
      issuance_quote_age_seconds: 5,
      decision_quote_age_seconds: 10,
      checks: [],
      supporting_checks: [],
      failed_checks: [],
      unavailable_checks: [],
      sizing: {},
      allocation: { state: "allocated", reason: "allocated", allocated_stake: 20 },
      suggested_stake: 20,
      bankroll_basis: null,
      portfolio_snapshot_id: null,
      portfolio_observed_at: null,
      offer_provenance: {
        issuance_id: "issuance-1",
        candidate_reference_id: "candidate-1",
        candidate_reference_derivation_version: 1,
        game_id: "2026_01_KC_LAC",
        market: "moneyline",
        side: "home",
        provider: "the_odds_api",
        sportsbook: "draftkings",
        fetched_at: "2026-09-01T12:00:00Z",
        is_live: false,
        american_price: -120,
      },
      forecast_provenance: {
        product_id: "product-1",
        product_run_id: "run-1",
        product_generated_at: "2026-09-01T12:00:00Z",
      },
      policy_provenance: {
        policy_id: "policy-1",
        policy_schema_version: 1,
        source_evidence_fingerprint: "e",
        governance_fingerprint: "g",
        derivation_method: "method",
      },
      ...overrides,
    };
  }

  function loadWithRecommendation(recommendation: Recommendation | null) {
    mockLoaded(detail());
    const base = edges();
    vi.mocked(useEdges).mockReturnValue({
      data: {
        ...base,
        diagnostics: {
          ...base.diagnostics,
          state: "positive_edges",
          positive_edge_count: 1,
          filtered_edge_count: 1,
        },
        items: [edgeWithRecommendation(recommendation)],
      },
      isLoading: false,
      error: null,
    } as never);
  }

  it("shows Top Analytical Edge and manual staging for no persisted result", () => {
    loadWithRecommendation(null);
    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getByText("Top Analytical Edge")).toBeInTheDocument();
    expect(screen.getByText("Add as manual wager")).toBeInTheDocument();
  });

  it("shows Persisted Policy Result and manual staging for a failed result", () => {
    loadWithRecommendation(
      persisted({
        decision_state: "unqualified",
        result_state: "failed",
        recommendation_eligible: false,
        allocation: { state: "not_evaluated", reason: "recommendation_ineligible", allocated_stake: null },
        suggested_stake: null,
      }),
    );
    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getByText("Persisted Policy Result")).toBeInTheDocument();
    expect(screen.getAllByText("Failed qualification").length).toBeGreaterThan(0);
    expect(screen.queryByText("Add recommendation to bet slip")).not.toBeInTheDocument();
    expect(screen.getByText("Add as manual wager")).toBeInTheDocument();
  });

  it("shows Governed Recommendation with allocation pending and manual staging", () => {
    loadWithRecommendation(
      persisted({
        result_state: "unavailable",
        allocation: { state: "not_evaluated", reason: "allocation_evidence_unavailable", allocated_stake: null },
        suggested_stake: null,
      }),
    );
    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getByText("Governed Recommendation")).toBeInTheDocument();
    expect(screen.getAllByText(/allocation pending/i).length).toBeGreaterThan(0);
    expect(screen.getByText("Add as manual wager")).toBeInTheDocument();
  });

  it("shows zero allocation with reason and manual staging", () => {
    loadWithRecommendation(
      persisted({
        allocation: { state: "zero_allocation", reason: "correlation_capacity_exhausted", allocated_stake: 0 },
        suggested_stake: null,
      }),
    );
    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getAllByText(/zero allocation/i).length).toBeGreaterThan(0);
    expect(screen.getByText("Correlated exposure capacity is exhausted.")).toBeInTheDocument();
    expect(screen.getByText("Add as manual wager")).toBeInTheDocument();
  });

  it("shows Recommended with positive stake and governed staging", () => {
    loadWithRecommendation(persisted({}));
    render(<TestWrapper><GameDetail /></TestWrapper>);

    expect(screen.getByText("Governed Recommendation")).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === "Recommended · $20.00"),
    ).toBeInTheDocument();
    expect(screen.getByText("Add recommendation to bet slip")).toBeInTheDocument();
  });
});
