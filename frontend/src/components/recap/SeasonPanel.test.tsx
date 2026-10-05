import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../../api/schema";
import { SeasonPanel } from "./SeasonPanel";
import { TestWrapper } from "../../test/testWrapper";

vi.mock("../../api/hooks", () => ({ useSeasonRecap: vi.fn() }));
import { useSeasonRecap } from "../../api/hooks";

type Season = components["schemas"]["SeasonRecapResponse"];
type WeekMetrics = components["schemas"]["RecapWeekMetrics"];
type Paired = components["schemas"]["RecapPairedDifference"];
type Bin = components["schemas"]["RecapReliabilityBin"];

const SEASON = "2026-2027";

function metrics(modelBrier: number, marketBrier: number, spreadMae = 12, count = 16): WeekMetrics {
  const errors = (mae: number) => ({ count, mae, rmse: mae + 2, bias: -0.5 });
  return {
    win: {
      model: { count, brier: modelBrier, log_loss: 0.66, accuracy: 0.625 },
      market: { count, brier: marketBrier, log_loss: 0.64, accuracy: 0.6875 },
      closer: { model: 7, market: 9, tie: 0 },
    },
    spread: { model: errors(spreadMae), market: errors(11), closer: { model: 6, market: 10, tie: 0 } },
    total: { model: errors(12.7), market: errors(11.7), closer: { model: 7, market: 9, tie: 0 } },
    projected_score: { model: errors(7.1), market: errors(6.6), closer: { model: 7, market: 9, tie: 0 } },
  };
}

const EMPTY = metrics(0, 0, 0, 0);
for (const side of ["model", "market"] as const) {
  EMPTY.win[side] = { count: 0, brier: null, log_loss: null, accuracy: null };
}

function bins(counts: number[]): Bin[] {
  return counts.map((count, index) => ({
    lower: index / 10, upper: (index + 1) / 10, count,
    mean_forecast: count ? index / 10 + 0.05 : null,
    observed_rate: count ? 0.5 : null,
  }));
}

function paired(count: number, mean: number | null, low: number | null, high: number | null): Paired {
  return { count, mean, standard_error: low == null ? null : 0.011, interval_low: low, interval_high: high };
}

function scope(weeks: number, games: number, season: WeekMetrics) {
  return {
    weeks, games, metrics: season,
    paired: {
      win: paired(games, 0.0083, -0.0138, 0.0305),
      spread: paired(games, 0.658, -0.153, 1.469),
      total: paired(1, 1.04, null, null),
      projected_score: paired(0, null, null, null),
    },
    calibration: {
      model: { count: games, ece: 0.1446, bins: bins([0, 0, 2, 7, 7, 11, 8, 10, 3, 0]) },
      market: { count: games, ece: 0.1283, bins: bins([0, 0, 2, 5, 9, 10, 16, 5, 1, 0]) },
    },
    starting_qbs: {
      slots: games * 2, scored: 0, from_capture: 0, not_recorded: games * 2, starter_unresolved: 0,
      not_in_product: 0, actual_missing: 0, actual_from_game_logs: 0,
      assumed: { count: 0, correct: 0, accuracy: null },
      nflverse: { count: 0, correct: 0, accuracy: null },
      carry_forward: { count: 0, correct: 0, accuracy: null },
    },
  };
}

function season(): Season {
  const week1 = metrics(0.2372, 0.2093);
  const week2 = metrics(0.2278, 0.2322);
  const toDate2 = metrics(0.2325, 0.2208, 12.0, 32);
  return {
    season: SEASON, season_recap_id: "s".repeat(64), built_at: "2026-10-05T18:00:00Z",
    selected_at: "2026-10-05T18:00:00Z", weekly_recap_ids: ["1".repeat(64), "2".repeat(64)],
    conventions: {
      recap: {
        orientation: "Away/Home", closing_market: "VEGAS_LINE and OVER_UNDER",
        market_at_forecast: "spread_line and total_line", market_win_probability: "Phi",
        margin_std: 13.5, actual_starter: "nflverse",
      },
      season_to_date: "pooled", live_only: "live rows",
      calibration_bins: "10 equal-width bins of the home win probability",
      paired_difference: "model minus market", interval: "mean +/- z se", interval_z: 1.96,
    },
    weeks: [
      { week: 1, status: "recapped", recap_id: "1".repeat(64), forecast_role: "live", games: 16, live_games: 16,
        metrics: week1, live_metrics: week1, to_date: week1, live_to_date: week1 },
      { week: 2, status: "recapped", recap_id: "2".repeat(64), forecast_role: "development", games: 16, live_games: 0,
        metrics: week2, live_metrics: EMPTY, to_date: toDate2, live_to_date: week1 },
      { week: 3, status: "not_built", recap_id: null, forecast_role: null, games: 0, live_games: 0,
        metrics: null, live_metrics: null, to_date: null, live_to_date: null },
    ],
    all_games: scope(2, 32, toDate2),
    live_games: scope(1, 16, week1),
  };
}

function renderPanel(data: Season | null = season()) {
  vi.mocked(useSeasonRecap).mockReturnValue({ data, isLoading: false, error: null } as never);
  render(<TestWrapper><SeasonPanel season={SEASON} /></TestWrapper>);
}

function seriesRow(series: string, row: string) {
  const figure = document.querySelector(`[data-season-series="${series}"]`) as HTMLElement;
  return figure.querySelector(`[data-season-row="${row}"]`) as HTMLElement;
}

describe("SeasonPanel", () => {
  beforeEach(() => vi.mocked(useSeasonRecap).mockReset());

  it("states that no season recap is selected", () => {
    renderPanel(null);
    expect(screen.getByText(`No season recap is selected for ${SEASON}`)).toBeInTheDocument();
  });

  it("shows each week's model beside the market and the season row", () => {
    renderPanel();
    const week1 = seriesRow("win", "week-1");
    expect(within(week1).getByText("0.237")).toBeInTheDocument();
    expect(within(week1).getByText("0.209")).toBeInTheDocument();
    expect(within(seriesRow("win", "week-2")).getByText("development")).toBeInTheDocument();
    expect(within(seriesRow("win", "season")).getByText("0.233")).toBeInTheDocument();
    expect(week1.querySelector(".season-dot-model")).not.toBeNull();
    expect(week1.querySelector(".season-dot-market")).not.toBeNull();
  });

  it("names a finished week without a recap instead of dropping it", () => {
    renderPanel();
    expect(within(seriesRow("win", "week-3")).getByText("Not built")).toBeInTheDocument();
    expect(screen.getByText("Week 3: finished, but no weekly recap is selected")).toBeInTheDocument();
  });

  it("switches to season-to-date values", () => {
    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Season to date" }));
    expect(within(seriesRow("win", "week-2")).getByText("0.233")).toBeInTheDocument();
    expect(within(seriesRow("win", "week-2")).getByText("0.221")).toBeInTheDocument();
  });

  it("keeps the live rows only and says when a week has none", () => {
    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Live only" }));
    expect(within(seriesRow("win", "week-2")).getByText("No live forecasts")).toBeInTheDocument();
    expect(within(seriesRow("win", "season")).getByText("0.237")).toBeInTheDocument();
    expect(screen.getByText(/1 week · 16 games · live forecasts only/)).toBeInTheDocument();
  });

  it("shows the paired difference with its interval, or why it has none", () => {
    renderPanel();
    const win = document.querySelector('[data-season-paired="win"]') as HTMLElement;
    expect(within(win).getByText("+0.0083")).toBeInTheDocument();
    expect(within(win).getByText(/\[-0.0138, \+0.0305\] n 32/)).toBeInTheDocument();
    expect(win.querySelector(".season-whisker")).not.toBeNull();
    const total = document.querySelector('[data-season-paired="total"]') as HTMLElement;
    expect(within(total).getByText(/an interval needs two games/)).toBeInTheDocument();
    expect(total.querySelector(".season-whisker")).toBeNull();
    const score = document.querySelector('[data-season-paired="projected_score"]') as HTMLElement;
    expect(within(score).getByText("No games both sides forecast")).toBeInTheDocument();
    expect(screen.getByText(/mean ± 1.96 × standard error, normal approximation/)).toBeInTheDocument();
  });

  it("plots reliability for non-empty bins only and tables every bin", () => {
    renderPanel();
    const figure = document.querySelector("[data-season-reliability]") as HTMLElement;
    expect(within(figure).getByText(/ECE model 0.145 \(32\) · market 0.128 \(32\)/)).toBeInTheDocument();
    expect(figure.querySelectorAll('[data-reliability-point="model"]')).toHaveLength(7);
    expect(figure.querySelectorAll('[data-reliability-point="market"]')).toHaveLength(7);
    expect(figure.querySelectorAll("[data-reliability-bin]")).toHaveLength(10);
    const empty = figure.querySelector('[data-reliability-bin="0"]') as HTMLElement;
    expect(within(empty).getAllByText("—")).toHaveLength(4);
  });

  it("states the season totals with counts", () => {
    renderPanel();
    const totals = screen.getByRole("table", { name: "Season totals, model and closing market" });
    const brier = within(totals).getByRole("row", { name: /Win Brier/ });
    expect(within(brier).getByText("0.2325 (32)")).toBeInTheDocument();
    expect(within(brier).getByText("0.2208 (32)")).toBeInTheDocument();
    expect(screen.getByText(/Starting QBs: 0 of 64 slots scored · 64 not recorded/)).toBeInTheDocument();
  });
});
