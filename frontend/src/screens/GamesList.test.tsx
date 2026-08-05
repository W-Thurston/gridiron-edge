import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { useGamesList } from "../api/hooks";
import { TestWrapper } from "../test/testWrapper";
import { GamesList } from "./GamesList";

vi.mock("../api/hooks", () => ({ useGamesList: vi.fn() }));
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
}

describe("GamesList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoaded([]);
  });

  it("keeps a scheduled row visible when Win is unavailable", () => {
    mockLoaded([
      game({
        win: { status: "forecast_missing" },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    expect(screen.getByText("Win forecast missing")).toBeInTheDocument();
    expect(screen.queryByText("No games found for this week.")).not.toBeInTheDocument();
    expect(screen.getByText("-2.5")).toBeInTheDocument();
  });

  it("renders a Total point estimate when uncertainty is unavailable", () => {
    mockLoaded([
      game({
        total: {
          status: "uncertainty_unavailable",
          model_total: 47.5,
        },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    expect(screen.getByText("47.5")).toHaveAttribute(
      "title",
      "Total available; uncertainty unavailable",
    );
  });

  it("shows exact Total unavailability", () => {
    mockLoaded([
      game({
        total: { status: "forecast_ambiguous" },
        projected_score: { status: "total_unavailable" },
      }),
    ]);

    render(<TestWrapper><GamesList /></TestWrapper>);

    expect(screen.getByText("Total forecast selection ambiguous")).toBeInTheDocument();
  });

  it("renders available Win probability", () => {
    mockLoaded([game()]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("58%")).toBeInTheDocument();
  });

  it("exposes an inconsistent available component with no value", () => {
    mockLoaded([
      game({ spread: { status: "available", model_spread: null } }),
    ]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("Value unavailable")).toHaveAccessibleName(
      "Spread has status available but no value.",
    );
  });

  it("renders schedule-empty copy only when there are no rows", () => {
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("No games found for this week.")).toBeInTheDocument();
  });

  it("renders canonical team marks for service-preserved long names", () => {
    mockLoaded([game()]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.getByText("KAN")).toBeInTheDocument();
    expect(screen.getByText("LAC")).toBeInTheDocument();
    expect(screen.queryByText("Kansas City Chiefs")).not.toBeInTheDocument();
  });

  it("does not restore Band or Confidence columns", () => {
    mockLoaded([game()]);
    render(<TestWrapper><GamesList /></TestWrapper>);
    expect(screen.queryByRole("columnheader", { name: "Band" })).not.toBeInTheDocument();
    expect(
      screen.queryByRole("columnheader", { name: "Confidence" }),
    ).not.toBeInTheDocument();
  });
});
