import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useComparables, useExplain, useGame } from "../api/hooks";
import type { components } from "../api/schema";
import { TestWrapper } from "../test/testWrapper";
import { ExplainPage } from "./ExplainPage";

vi.mock("../api/hooks", () => ({
  useGame: vi.fn(),
  useExplain: vi.fn(),
  useComparables: vi.fn(),
}));
vi.mock("../context/NavContext", () => ({
  useNav: vi.fn(() => ({
    route: { path: "/explain", params: { gameId: "2026_02_CAR_ATL" } },
    navigate: vi.fn(),
  })),
  NavProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

type GameDetail = components["schemas"]["GameDetail"];
type GameExplain = components["schemas"]["GameExplain"];
type GameComparables = components["schemas"]["GameComparables"];
type ExplainFactor = components["schemas"]["ExplainFactor"];
type ComparableGame = components["schemas"]["ComparableGame"];

function game(overrides: Partial<GameDetail> = {}): GameDetail {
  return {
    game_id: "2026_02_CAR_ATL",
    game_date: "2026-09-14",
    away_team: "Carolina Panthers",
    home_team: "Atlanta Falcons",
    win: { status: "available" },
    spread: { status: "available" },
    total: { status: "available" },
    projected_score: { status: "available" },
    ...overrides,
  } as GameDetail;
}

function factor(overrides: Partial<ExplainFactor> = {}): ExplainFactor {
  return {
    key: "ELO_DIFF",
    label: "ELO_DIFF",
    log_odds_contribution: 0.1,
    coefficient: 0.5,
    transformed_value: 0.2,
    is_baseline: false,
    is_adjustable: false,
    ...overrides,
  };
}

function comparable(overrides: Partial<ComparableGame> = {}): ComparableGame {
  return {
    game_id: "2006_19_DAL_SEA",
    rank: 1,
    distance: 8.59,
    season: "2006-2007",
    week: 19,
    game_date: "2007-01-06",
    away_team: "Dallas Cowboys",
    home_team: "Seattle Seahawks",
    away_score: 20,
    home_score: 21,
    favorite_team: "Seattle Seahawks",
    spread_magnitude: 1.0,
    favorite_won: true,
    favorite_covered: false,
    top_contributing_features: [],
    ...overrides,
  };
}

function mockLoaded({
  explain,
  comparables,
}: {
  explain: GameExplain;
  comparables: GameComparables;
}) {
  vi.mocked(useGame).mockReturnValue({
    data: game(),
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useExplain).mockReturnValue({
    data: explain,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useComparables).mockReturnValue({
    data: comparables,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

describe("ExplainPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows loading state while any query is in flight", () => {
    vi.mocked(useGame).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as never);
    vi.mocked(useExplain).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as never);
    vi.mocked(useComparables).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as never);

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders the headline win probability and real factors", () => {
    mockLoaded({
      explain: {
        game_id: "2026_02_CAR_ATL",
        headline_win_prob: 0.5541,
        factors: [
          factor({ key: "intercept", label: "intercept", is_baseline: true, log_odds_contribution: 0.3 }),
          factor({ key: "ELO_DIFF", label: "ELO_DIFF", log_odds_contribution: 0.13 }),
        ],
      },
      comparables: { game_id: "2026_02_CAR_ATL", comparables: [comparable()], sample_size: 1 },
    });

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    expect(screen.getByText("55.4%")).toBeInTheDocument();
    expect(screen.getByText("intercept")).toBeInTheDocument();
    expect(screen.getByText("ELO_DIFF")).toBeInTheDocument();
    expect(screen.getByText("+0.130")).toBeInTheDocument();
  });

  it("sorts factors by magnitude and collapses beyond the top 8", () => {
    const rest = Array.from({ length: 10 }, (_, index) =>
      factor({
        key: `FEATURE_${index}`,
        label: `FEATURE_${index}`,
        log_odds_contribution: (index + 1) * 0.01,
      }),
    );
    mockLoaded({
      explain: {
        game_id: "2026_02_CAR_ATL",
        headline_win_prob: 0.5,
        factors: [
          factor({ key: "intercept", label: "intercept", is_baseline: true }),
          ...rest,
        ],
      },
      comparables: { game_id: "2026_02_CAR_ATL", comparables: [], sample_size: 0 },
    });

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    // Largest magnitude (FEATURE_9, contribution 0.10) shown; smallest two hidden.
    expect(screen.getByText("FEATURE_9")).toBeInTheDocument();
    expect(screen.queryByText("FEATURE_0")).not.toBeInTheDocument();
    expect(screen.getByText("Show 2 more factors")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Show 2 more factors"));

    expect(screen.getByText("FEATURE_0")).toBeInTheDocument();
    expect(screen.getByText("Show fewer factors")).toBeInTheDocument();
  });

  it("shows a blocked placeholder when factors are unavailable, not a crash", () => {
    mockLoaded({
      explain: {
        game_id: "2026_02_CAR_ATL",
        headline_win_prob: 0.62,
        factors: null,
        _meta: {
          field_status: {
            factors: {
              status: "blocked",
              blocker: "feature_attribution",
              roadmap: "deferred",
            },
          },
        },
      },
      comparables: { game_id: "2026_02_CAR_ATL", comparables: null, sample_size: null },
    });

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    expect(screen.getByText("62.0%")).toBeInTheDocument();
    expect(screen.getAllByTitle(/feature_attribution/).length).toBeGreaterThan(0);
  });

  it("distinguishes an honestly-empty comparables result from a blocked one", () => {
    mockLoaded({
      explain: {
        game_id: "2026_02_CAR_ATL",
        headline_win_prob: 0.5,
        factors: [factor({ key: "intercept", label: "intercept", is_baseline: true })],
      },
      comparables: { game_id: "2026_02_CAR_ATL", comparables: [], sample_size: 0 },
    });

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    expect(
      screen.getByText(/No historical game was similar enough/),
    ).toBeInTheDocument();
  });

  it("renders a real comparable game's outcome", () => {
    mockLoaded({
      explain: {
        game_id: "2026_02_CAR_ATL",
        headline_win_prob: 0.5,
        factors: [factor({ key: "intercept", label: "intercept", is_baseline: true })],
      },
      comparables: {
        game_id: "2026_02_CAR_ATL",
        comparables: [comparable()],
        sample_size: 1,
        favorite_win_rate: 1.0,
        favorite_cover_rate: 0.0,
      },
    });

    render(
      <TestWrapper>
        <ExplainPage />
      </TestWrapper>,
    );

    expect(screen.getByText("20–21")).toBeInTheDocument();
    expect(screen.getByText("Won")).toBeInTheDocument();
    expect(screen.getByText("No cover")).toBeInTheDocument();
    expect(screen.getByText(/Favorite: Seattle Seahawks/)).toBeInTheDocument();
  });
});
