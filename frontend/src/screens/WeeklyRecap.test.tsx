import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { WeeklyRecap } from "./WeeklyRecap";
import { TestWrapper } from "../test/testWrapper";

vi.mock("../api/hooks", () => ({
  useRecaps: vi.fn(),
  useWeeklyRecap: vi.fn(),
}));
import { useRecaps, useWeeklyRecap } from "../api/hooks";

type Recap = components["schemas"]["WeeklyRecapResponse"];
type RecapGame = components["schemas"]["RecapGame"];
type Summary = components["schemas"]["WeeklyRecapSummary"];

const SEASON = "2026-2027";

function summary(week: number, role: Summary["forecast_role"], builtAt: string): Summary {
  return {
    season: SEASON, week, recap_id: `${week}`.repeat(64), built_at: builtAt, selected_at: builtAt,
    product_id: `product-${week}`, forecast_role: role, row_forecast_roles: role ? [role] : [],
    scheduled_games: 4, in_product: 3, available: { win: 2, spread: 3, total: 2, projected_score: 2 },
  };
}

function game(id: string, overrides: Partial<RecapGame> = {}): RecapGame {
  return {
    game_id: id, away_team: "AAA", home_team: "BBB", kickoff: "2026-10-04T17:00:00Z",
    product_row_status: "in_product",
    forecast: { role: "development", run_id: "run", generated_at: "2026-10-01T12:00:00Z", carried: false },
    result: { away_score: 17, home_score: 24, home_margin: 7, total: 41, home_win: true },
    market_at_forecast: { status: "not_recorded", home_margin: null, total: null, ingested_at: null },
    closing_market: {
      spread_status: "available", home_margin: 2.5, home_win_prob: 0.574, total_status: "available",
      total: 45.5, implied_away_score: 21.5, implied_home_score: 24,
    },
    win: { status: "available", home_win_prob: 0.612, model_brier: 0.1505, market_brier: 0.1815, closer: "model" },
    spread: { status: "available", value: 3.4, model_error: -3.6, market_error: -4.5, closer: "model" },
    total: { status: "available", value: 44.2, model_error: 3.2, market_error: 4.5, closer: "model" },
    projected_score: {
      status: "available", away_score: 20.4, home_score: 23.8, model_abs_error: 1.8,
      market_abs_error: 2.25, closer: "model",
    },
    ...overrides,
  };
}

const notInProduct = game("2026_02_EEE_FFF", {
  away_team: "EEE", home_team: "FFF", product_row_status: "not_in_product",
  forecast: { role: null, run_id: null, generated_at: null, carried: null },
  market_at_forecast: { status: "not_in_product", home_margin: null, total: null, ingested_at: null },
  win: { status: "not_in_product", home_win_prob: null, model_brier: null, market_brier: 0.2, closer: null },
  spread: { status: "not_in_product", value: null, model_error: null, market_error: 2, closer: null },
  total: { status: "not_in_product", value: null, model_error: null, market_error: 7, closer: null },
  projected_score: {
    status: "not_in_product", away_score: null, home_score: null, model_abs_error: null,
    market_abs_error: 3.5, closer: null,
  },
});

const tied = game("2026_02_CCC_DDD", {
  away_team: "CCC", home_team: "DDD",
  forecast: { role: "development", run_id: "earlier", generated_at: "2026-09-30T12:00:00Z", carried: true },
  result: { away_score: 20, home_score: 20, home_margin: 0, total: 40, home_win: null },
  win: { status: "outcome_tie", home_win_prob: 0.55, model_brier: null, market_brier: null, closer: null },
});

const totalMissing = game("2026_02_GGG_HHH", {
  away_team: "GGG", home_team: "HHH",
  closing_market: {
    spread_status: "closing_line_missing", home_margin: null, home_win_prob: null,
    total_status: "closing_line_missing", total: null, implied_away_score: null, implied_home_score: null,
  },
  win: { status: "available", home_win_prob: 0.6, model_brier: 0.16, market_brier: null, closer: null },
  spread: { status: "available", value: 2, model_error: -1, market_error: null, closer: null },
  total: { status: "forecast_missing", value: null, model_error: null, market_error: null, closer: null },
  projected_score: {
    status: "forecast_missing", away_score: null, home_score: null, model_abs_error: null,
    market_abs_error: null, closer: null,
  },
});

function recap(week: number): Recap {
  const probability = { count: 2, brier: 0.1555, log_loss: 0.4912, accuracy: 1 };
  const errors = { count: 3, mae: 2.6, rmse: 2.9, bias: -1.2 };
  return {
    season: SEASON, week, recap_id: "a".repeat(64), built_at: "2026-10-05T01:30:00Z",
    selected_at: "2026-10-05T01:30:00Z", product_id: `product-${week}`, product_run_id: "run",
    carried_run_ids: ["earlier"], product_schema_version: 1, forecast_role: "development",
    conventions: {
      orientation: "Away/Home", closing_market: "VEGAS_LINE and OVER_UNDER",
      market_at_forecast: "spread_line and total_line", market_win_probability: "Phi(VEGAS_LINE / margin_std)",
      margin_std: 13.5,
    },
    counts: { scheduled_games: 4, in_product: 3, not_in_product: 1, carried_rows: 1, outcome_ties: 1 },
    metrics: {
      win: { model: probability, market: { ...probability, brier: 0.1815 }, closer: { model: 1, market: 0, tie: 0 } },
      spread: { model: errors, market: { ...errors, mae: 3.1 }, closer: { model: 2, market: 1, tie: 0 } },
      total: { model: errors, market: errors, closer: { model: 1, market: 1, tie: 0 } },
      projected_score: { model: { ...errors, count: 4 }, market: errors, closer: { model: 1, market: 1, tie: 0 } },
    },
    games: [game("2026_02_AAA_BBB"), tied, notInProduct, totalMissing],
    starting_qbs: null, edges: null, wagers: null, season_to_date: null,
    _meta: {
      field_status: { starting_qbs: "pending", edges: "pending", wagers: "pending", season_to_date: "pending" },
    },
  };
}

function loaded(items: Summary[]) {
  vi.mocked(useRecaps).mockReturnValue({ data: { items }, isLoading: false, error: null } as never);
  vi.mocked(useWeeklyRecap).mockImplementation(((season: string | null, week: number | null) => ({
    data: season && week ? recap(week) : undefined, isLoading: false, error: null,
  })) as never);
}

function renderScreen() {
  render(<TestWrapper><WeeklyRecap /></TestWrapper>);
}

describe("WeeklyRecap", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "#/recap");
    sessionStorage.clear();
    vi.mocked(useWeeklyRecap).mockReset();
  });

  it("defaults to the highest week, not the latest build", () => {
    loaded([
      summary(1, "live", "2026-10-06T00:00:00Z"),
      summary(2, "development", "2026-10-05T00:00:00Z"),
    ]);
    renderScreen();
    expect(useWeeklyRecap).toHaveBeenLastCalledWith(SEASON, 2);
    expect(screen.getByRole("heading", { name: "2026-2027 week 2" })).toBeInTheDocument();
  });

  it("names a development forecast explicitly", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    expect(
      screen.getByText("Development forecast: regenerated after the fact, not issued before kickoff"),
    ).toBeInTheDocument();
  });

  it("shows model beside market for every family", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const win = document.querySelector('[data-recap-family="win"]') as HTMLElement;
    expect(within(win).getByText("0.1555")).toBeInTheDocument();
    expect(within(win).getByText("0.1815")).toBeInTheDocument();
    expect(within(win).getByText(/model 1 · market 0 · tie 0/)).toBeInTheDocument();
    for (const family of ["spread", "total", "projected_score"]) {
      expect(document.querySelector(`[data-recap-family="${family}"]`)).not.toBeNull();
    }
    expect(screen.getByText(/Φ\(closing home margin \/ 13.5\)/)).toBeInTheDocument();
  });

  it("keeps a game the product lacks as a row with an explicit status", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const row = document.querySelector('[data-product-row-status="not_in_product"]') as HTMLElement;
    expect(within(row).getByText("EEE @ FFF")).toBeInTheDocument();
    expect(within(row).getByText("No forecast in the selected product")).toBeInTheDocument();
    expect(within(row).getByText("57.4%")).toBeInTheDocument();
  });

  it("explains a tie, a carried row, a missing forecast and a missing closing line", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    expect(screen.getAllByText("Tied game").length).toBeGreaterThan(0);
    expect(screen.getByText("55.0%")).toBeInTheDocument();
    expect(screen.getByText("carried from an earlier run")).toBeInTheDocument();
    expect(screen.getAllByText("No closing line").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Total" }));
    expect(screen.getByText("Total forecast missing from the selected product")).toBeInTheDocument();
    expect(screen.getAllByText("Market at forecast time not recorded").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Projected score" }));
    expect(screen.getByText("Projected score forecast missing from the selected product")).toBeInTheDocument();
    expect(screen.getAllByText("20.4 – 23.8").length).toBeGreaterThan(0);
  });

  it("switches weeks through the route", () => {
    loaded([
      summary(1, "live", "2026-10-05T00:00:00Z"),
      summary(2, "development", "2026-10-05T00:00:00Z"),
    ]);
    renderScreen();
    fireEvent.click(screen.getByRole("button", { name: "Week 1" }));
    expect(useWeeklyRecap).toHaveBeenLastCalledWith(SEASON, 1);
    expect(window.location.hash).toBe("#/recap?season=2026-2027&week=1");
  });

  it("links each game to its game page", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    fireEvent.click(screen.getByRole("button", { name: "AAA @ BBB" }));
    expect(window.location.hash).toBe("#/games?gameId=2026_02_AAA_BBB");
  });

  it("renders the sections the recap does not carry yet as pending", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    for (const key of ["starting_qbs", "edges", "wagers", "season_to_date"]) {
      const card = document.querySelector(`[data-recap-section="${key}"]`) as HTMLElement;
      expect(within(card).getByTitle("Coming soon")).toBeInTheDocument();
    }
  });

  it("explains how to build a recap when none is selected", () => {
    loaded([]);
    renderScreen();
    expect(screen.getByText("No weekly recap has been built yet")).toBeInTheDocument();
    expect(useWeeklyRecap).toHaveBeenLastCalledWith(null, null);
  });
});
