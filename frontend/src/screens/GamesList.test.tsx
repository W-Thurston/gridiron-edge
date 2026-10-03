import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { useGamesList, useLines } from "../api/hooks";
import { TestWrapper } from "../test/testWrapper";
import { GamesList } from "./GamesList";

vi.mock("../api/hooks", () => ({ useGamesList: vi.fn(), useLines: vi.fn() }));
vi.mock("../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn((identity: string) => {
    if (identity === "Kansas City Chiefs") {
      return {
        abbr: "KAN",
        name: "Kansas City Chiefs",
        primary_color: "#E31837",
      };
    }
    if (identity === "Los Angeles Chargers") {
      return {
        abbr: "LAC",
        name: "Los Angeles Chargers",
        primary_color: "#0080C6",
      };
    }
    return null;
  }),
}));

type GameList = components["schemas"]["GameList"];
type GameSummary = components["schemas"]["GameSummary"];

function game(overrides: Partial<GameSummary> = {}): GameSummary {
  return {
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
    starting_qb: { status: "not_recorded" },
    ...overrides,
  };
}

function response(items: GameSummary[]): GameList {
  return {
    season: "2026-2027",
    week: 1,
    total: items.length,
    items,
  };
}

function mockLoaded(items: GameSummary[]) {
  vi.mocked(useGamesList).mockReturnValue({
    data: response(items),
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useLines).mockReturnValue({
    data: {
      season: "2026-2027",
      week: 1,
      market: "moneyline",
      total: 0,
      items: [],
      sportsbooks: [],
      market_fetched_at: [],
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

describe("GamesList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "#/games?market=moneyline");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    mockLoaded([]);
  });

  it("keeps a scheduled row visible when Win is unavailable", () => {
    mockLoaded([
      game({
        win: { status: "forecast_missing" },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    expect(screen.getAllByText("Win forecast missing")).toHaveLength(2);
    expect(screen.queryByText("No games found for this week.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Details →" })).toBeInTheDocument();
  });

  it("renders a Total point estimate when uncertainty is unavailable", () => {
    window.history.replaceState(null, "", "#/games?market=total");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    mockLoaded([
      game({
        total: {
          status: "uncertainty_unavailable",
          model_total: 47.5,
        },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    const values = screen.getAllByText("47.5");
    expect(values).toHaveLength(2);
    for (const value of values) {
      expect(value).toHaveAttribute(
        "title",
        "Total available; uncertainty unavailable",
      );
    }
  });

  it("shows exact Total unavailability", () => {
    window.history.replaceState(null, "", "#/games?market=total");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    mockLoaded([
      game({
        total: { status: "forecast_ambiguous" },
        projected_score: { status: "total_unavailable" },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    expect(screen.getAllByText("Total forecast selection ambiguous")).toHaveLength(2);
  });

  it("renders available Win probability", () => {
    mockLoaded([game()]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("58%")).toBeInTheDocument();
  });

  it("exposes an inconsistent available component with no value", () => {
    window.history.replaceState(null, "", "#/games?market=spread");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    mockLoaded([
      game({ spread: { status: "available", model_spread: null } }),
    ]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getAllByText("Value unavailable")).toHaveLength(2);
    expect(screen.getByLabelText(
      "Away spread has status available but no value.",
    )).toBeInTheDocument();
    expect(screen.getByLabelText(
      "Home spread has status available but no value.",
    )).toBeInTheDocument();
  });

  it("renders schedule-empty copy only when there are no rows", () => {
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("No games found for this week.")).toBeInTheDocument();
  });

  it("renders canonical team marks for service-preserved long names", () => {
    mockLoaded([game()]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByLabelText("Kansas City Chiefs at Los Angeles Chargers")).toBeInTheDocument();
    expect(screen.getByLabelText("Kansas City Chiefs team mark")).toHaveTextContent("KAN");
    expect(screen.getByLabelText("Los Angeles Chargers team mark")).toHaveTextContent("LAC");
    expect(screen.getByText("Chiefs")).toBeInTheDocument();
    expect(screen.getByText("Chargers")).toBeInTheDocument();
    expect(screen.queryByText("Away team")).not.toBeInTheDocument();
    expect(screen.queryByText("Home team")).not.toBeInTheDocument();
  });

});
