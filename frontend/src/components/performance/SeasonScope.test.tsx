import { useState } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { components } from "../../api/schema";
import { SeasonNotBuilt, SeasonScope } from "./SeasonScope";
import type { SeasonState } from "./SeasonScope";
import { SEASON_DEFAULT_MODELS, seasonComparison } from "./compareData";
import { TestWrapper } from "../../test/testWrapper";

type Season = components["schemas"]["SeasonRecapResponse"];
type WeekMetrics = components["schemas"]["RecapWeekMetrics"];
type Paired = components["schemas"]["RecapPairedDifference"];
type Bin = components["schemas"]["RecapReliabilityBin"];
type WagerSummary = components["schemas"]["RecapWagerSummary"];

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

function wagers(bets: number, wins: number, losses: number, pnl: number, unsettled = 0): WagerSummary {
  const settled = wins + losses;
  return {
    bets, settled, unsettled, wins, losses, pushes: 0, staked: bets, settled_staked: settled, pnl,
    roi_pct: settled ? (pnl / settled) * 100 : null, bonus_funded: 0, clv_available: 0,
    clv_unavailable: bets, linked_recommendations: 0,
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
      unattributed_wagers: "bets without a game placed in the season window",
    },
    weeks: [
      { week: 1, status: "recapped", recap_id: "1".repeat(64), forecast_role: "live", games: 16, live_games: 16,
        metrics: week1, live_metrics: week1, to_date: week1, live_to_date: week1,
        edge_status: "production_chain", wagers: wagers(17, 11, 6, -0.34) },
      { week: 2, status: "recapped", recap_id: "2".repeat(64), forecast_role: "development", games: 16, live_games: 0,
        metrics: week2, live_metrics: EMPTY, to_date: toDate2, live_to_date: week1,
        edge_status: "not_recorded", wagers: wagers(16, 12, 4, 2.53) },
      { week: 3, status: "not_built", recap_id: null, forecast_role: null, games: 0, live_games: 0,
        metrics: null, live_metrics: null, to_date: null, live_to_date: null,
        edge_status: null, wagers: null },
    ],
    all_games: scope(2, 32, toDate2),
    live_games: scope(1, 16, week1),
    wagers: {
      recapped_weeks: wagers(33, 23, 10, 2.19),
      unattributed: wagers(16, 0, 16, -5.2),
      unattributed_digest: "u".repeat(64),
    },
  };
}

/** Holds the scope's state the way the page does through the route. */
function Harness({ recap, onOpenWeek }: { recap: Season; onOpenWeek: (week: number) => void }) {
  const [state, setState] = useState<SeasonState>({ games: "all", view: "week", models: SEASON_DEFAULT_MODELS, metric: undefined });
  return <SeasonScope recap={recap} state={state} onChange={(next) => setState({ ...state, ...next })} onOpenWeek={onOpenWeek} />;
}

function renderScope(onOpenWeek = vi.fn()) {
  render(<TestWrapper><Harness recap={season()} onOpenWeek={onOpenWeek} /></TestWrapper>);
  return onOpenWeek;
}

function weekRow(week: number) {
  const table = screen.getByRole("region", { name: "Weekly recaps" });
  return table.querySelector(`[data-season-row="week-${week}"]`) as HTMLElement;
}

describe("SeasonScope", () => {
  it("states that no season recap is selected", () => {
    render(<TestWrapper><SeasonNotBuilt season={SEASON} /></TestWrapper>);
    expect(screen.getByText(`No season recap is selected for ${SEASON}`)).toBeInTheDocument();
    expect(screen.getByText(/--season-only/)).toBeInTheDocument();
  });

  it("titles the season and counts what it covers", () => {
    renderScope();
    expect(screen.getByRole("heading", { level: 1, name: "2026-2027 season · weeks 1–2" })).toBeInTheDocument();
    expect(screen.getByText("Includes development weeks")).toBeInTheDocument();
    const counts = screen.getByRole("list", { name: "Season counts" });
    expect(within(counts.querySelector('[data-count="Weeks recapped"]') as HTMLElement).getByText("2")).toBeInTheDocument();
    expect(within(counts.querySelector('[data-count="Weeks not built"]') as HTMLElement).getByText("1")).toBeInTheDocument();
  });

  it("shows each week's model beside the market and how often the model was closer", () => {
    renderScope();
    const week1 = weekRow(1);
    expect(within(week1).getByText("0.237")).toBeInTheDocument();
    expect(within(week1).getByText("0.209")).toBeInTheDocument();
    expect(within(week1).getAllByText("7/16").length).toBe(2);
    expect(within(week1).getByText("6/16")).toBeInTheDocument();
    expect(within(weekRow(2)).getByText("development")).toBeInTheDocument();
  });

  it("opens a week from its row", () => {
    const onOpenWeek = renderScope();
    fireEvent.click(within(weekRow(2)).getByRole("button", { name: "Week 2" }));
    expect(onOpenWeek).toHaveBeenCalledWith(2);
    fireEvent.click(weekRow(1));
    expect(onOpenWeek).toHaveBeenLastCalledWith(1);
  });

  it("names a finished week without a recap instead of dropping it", () => {
    const onOpenWeek = renderScope();
    expect(within(weekRow(3)).getByText("Not built")).toHaveAttribute("data-weekly-status", "not_built");
    expect(screen.getByText("Week 3: finished, but no weekly recap is selected")).toBeInTheDocument();
    fireEvent.click(weekRow(3));
    expect(onOpenWeek).not.toHaveBeenCalled();
  });

  it("compares the published model with the closing line week by week or to date", () => {
    const recap = season();
    expect(seasonComparison(recap, "all", "week").values.win_brier).toEqual({
      champion: [0.2372, 0.2278, null], closing_line: [0.2093, 0.2322, null],
    });
    expect(seasonComparison(recap, "all", "to_date").values.win_brier.champion).toEqual([0.2372, 0.2325, null]);
    expect(seasonComparison(recap, "live", "week").values.win_brier.champion).toEqual([0.2372, null, null]);
    expect(seasonComparison(recap, "all", "week").periods).toEqual(["W1", "W2", "W3"]);

    renderScope();
    fireEvent.click(screen.getByRole("button", { name: "To date" }));
    expect(screen.getByText("Compare · season to date")).toBeInTheDocument();
  });

  it("keeps the live rows only and says when a week has none", () => {
    renderScope();
    fireEvent.click(screen.getByRole("button", { name: "Live only" }));
    expect(within(weekRow(2)).getByText("No live forecasts")).toBeInTheDocument();
    expect(screen.getByText(/1 week · 16 games · live forecasts only/)).toBeInTheDocument();
    expect(screen.getByText("Live forecasts only")).toBeInTheDocument();
    const win = document.querySelector('[data-recap-family="win"]') as HTMLElement;
    expect(within(win).getByText("0.2372")).toBeInTheDocument();
  });

  it("states the pooled season scorecard with counts", () => {
    renderScope();
    const win = document.querySelector('[data-recap-family="win"]') as HTMLElement;
    expect(within(win).getByText("0.2325")).toBeInTheDocument();
    expect(within(win).getByText("0.2208")).toBeInTheDocument();
    expect(within(win).getByText("Market ahead")).toBeInTheDocument();
    expect(within(within(win).getByRole("row", { name: /Games/ })).getAllByText("32")).toHaveLength(2);
    expect(screen.getByText(/Starting QBs: 0 of 64 slots scored · 64 not recorded/)).toBeInTheDocument();
  });

  it("shows the paired difference with its interval, or why it has none", () => {
    renderScope();
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
    renderScope();
    const figure = document.querySelector("[data-season-reliability]") as HTMLElement;
    expect(within(figure).getByText(/ECE model 0.145 \(32\) · market 0.128 \(32\)/)).toBeInTheDocument();
    expect(figure.querySelectorAll('[data-reliability-point="model"]')).toHaveLength(7);
    expect(figure.querySelectorAll('[data-reliability-point="market"]')).toHaveLength(7);
    expect(figure.querySelectorAll("[data-reliability-bin]")).toHaveLength(10);
    const empty = figure.querySelector('[data-reliability-bin="0"]') as HTMLElement;
    expect(within(empty).getAllByText("—")).toHaveLength(4);
  });

  it("lists each week's edges and wagers, the season total and bets without a game", () => {
    renderScope();
    const table = screen.getByRole("table", { name: /Each week's edge evidence and recorded wagers/ });
    const week1 = table.querySelector('[data-season-wager-week="1"]') as HTMLElement;
    expect(within(week1).getByText("Production chain")).toBeInTheDocument();
    expect(within(week1).getByText("11-6-0")).toBeInTheDocument();
    expect(within(week1).getByText("−$0.34")).toBeInTheDocument();
    const week3 = table.querySelector('[data-season-wager-week="3"]') as HTMLElement;
    expect(within(week3).getByText("No weekly recap is selected")).toHaveAttribute("data-weekly-status", "not_built");
    const total = table.querySelector('[data-season-wager-week="total"]') as HTMLElement;
    expect(within(total).getByText("+$2.19")).toBeInTheDocument();
    const unattributed = table.querySelector('[data-season-wager-week="unattributed"]') as HTMLElement;
    expect(within(unattributed).getByText("0-16-0")).toBeInTheDocument();
    expect(within(unattributed).getByText("$16.00")).toBeInTheDocument();
    expect(screen.getByText(/Without a game: bets without a game placed in the season window/)).toBeInTheDocument();
  });
});
