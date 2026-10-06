import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { WeeklyRecap } from "./WeeklyRecap";
import { TestWrapper } from "../test/testWrapper";

vi.mock("../api/hooks", () => ({
  useRecaps: vi.fn(),
  useWeeklyRecap: vi.fn(),
  useSeasonRecap: vi.fn(),
}));
import { useRecaps, useSeasonRecap, useWeeklyRecap } from "../api/hooks";

type Recap = components["schemas"]["WeeklyRecapResponse"];
type RecapGame = components["schemas"]["RecapGame"];
type Summary = components["schemas"]["WeeklyRecapSummary"];
type StarterSide = components["schemas"]["RecapStarterSide"];
type WagerSummary = components["schemas"]["RecapWagerSummary"];

const SEASON = "2026-2027";

function summary(week: number, role: Summary["forecast_role"], builtAt: string): Summary {
  return {
    season: SEASON, week, recap_id: `${week}`.repeat(64), built_at: builtAt, selected_at: builtAt,
    product_id: `product-${week}`, forecast_role: role, row_forecast_roles: role ? [role] : [],
    scheduled_games: 4, in_product: 3, available: { win: 2, spread: 3, total: 2, projected_score: 2 },
  };
}

function starter(overrides: Partial<StarterSide> = {}): StarterSide {
  return {
    status: "available", record_source: "product_record",
    assumed_id: "q1", assumed_name: "Right Starter", assumed_source: "depth_chart_report",
    assumed_as_of: "2026-10-03T12:00:00Z", nflverse_id: "q1", nflverse_name: "Right Starter",
    carried_forward_id: "q1", carried_forward_name: "Right Starter",
    actual_id: "q1", actual_name: "Right Starter", actual_source: "nflverse_schedule",
    assumed_correct: true, nflverse_correct: true, carry_forward_correct: true,
    ...overrides,
  };
}

const UNSCORED = {
  record_source: null, assumed_id: null, assumed_name: null, assumed_source: null, assumed_as_of: null,
  nflverse_id: null, nflverse_name: null, carried_forward_id: null, carried_forward_name: null,
  assumed_correct: null, nflverse_correct: null, carry_forward_correct: null,
} as const;

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
    starting_qbs: {
      away: starter(),
      home: starter({
        assumed_id: "q9", assumed_name: "Wrong Rookie", assumed_correct: false,
        actual_name: "Home Veteran", nflverse_name: "Home Veteran", carried_forward_name: "Home Veteran",
      }),
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
  starting_qbs: {
    away: starter({ ...UNSCORED, status: "not_in_product" }),
    home: starter({ ...UNSCORED, status: "not_in_product" }),
  },
});

const tied = game("2026_02_CCC_DDD", {
  away_team: "CCC", home_team: "DDD",
  forecast: { role: "development", run_id: "earlier", generated_at: "2026-09-30T12:00:00Z", carried: true },
  result: { away_score: 20, home_score: 20, home_margin: 0, total: 40, home_win: null },
  win: { status: "outcome_tie", home_win_prob: 0.55, model_brier: null, market_brier: null, closer: null },
  starting_qbs: {
    away: starter({
      record_source: "sidecar_capture", assumed_name: "Projected Starter", nflverse_id: "q0",
      nflverse_name: "Stale Listing", nflverse_correct: false, carried_forward_id: "q0",
      carried_forward_name: "Stale Listing", carry_forward_correct: false,
      actual_name: "Projected Starter", actual_source: "game_log_top_passer",
    }),
    home: starter({ record_source: "sidecar_capture" }),
  },
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
  starting_qbs: {
    away: starter({ ...UNSCORED, status: "not_recorded" }),
    home: starter({ ...UNSCORED, status: "not_recorded" }),
  },
});

const WITHHELD = "edge calculation blocked: model_edges_unvalidated";

function wagerSummary(overrides: Partial<WagerSummary> = {}): WagerSummary {
  return {
    bets: 17, settled: 16, unsettled: 1, wins: 10, losses: 6, pushes: 0, staked: 16.1,
    settled_staked: 15.1, pnl: -0.44, roi_pct: -2.9, bonus_funded: 1, clv_available: 0,
    clv_unavailable: 17, linked_recommendations: 0, ...overrides,
  };
}

/** Week 1 is a production-chain week; week 2 withheld after publishing; week 3 has nothing. */
function edges(week: number): Recap["edges"] {
  const receiptsThrough = "2026-10-06T00:15:00Z";
  if (week === 1) {
    return {
      status: "production_chain", receipts_through: receiptsThrough, receipts: [],
      chain: {
        issuance_id: "e945987f".padEnd(64, "0"), issued_at: "2026-08-27T23:34:46Z", rows: 1680,
        candidates: 698, not_candidates: 982, unavailable: 0,
        evaluation_ids: ["100a70a8".padEnd(64, "0")], policy_ids: ["p".repeat(64)],
        recommendations: [{ status: "unavailable", count: 698 }],
        families: [
          {
            market: "moneyline", candidates: 228, sides: 14,
            closeout: [{ status: "available", count: 118 }, { status: "kickoff_conflict", count: 104 }],
            clv_kind: "moneyline_price", clv_count: 118, median_clv: 0.0637, share_positive_clv: 0.7458,
            wins: 110, losses: 118, pushes: 0, outcome_unavailable: 0,
          },
          {
            market: "spread", candidates: 0, sides: 0, closeout: [], clv_kind: null, clv_count: 0,
            median_clv: null, share_positive_clv: null, wins: 0, losses: 0, pushes: 0, outcome_unavailable: 0,
          },
        ],
      },
    };
  }
  if (week === 2) {
    return {
      status: "withheld_unvalidated", chain: null, receipts_through: receiptsThrough,
      receipts: [
        { decided_at: "2026-10-04T20:00:54Z", outcome: "done", detail: "149 edges (top EV 27.1%)",
          product_id: "product-2", market_fetched_at: "2026-10-04T19:29:18Z" },
        { decided_at: "2026-10-04T22:00:54Z", outcome: "blocked", detail: WITHHELD,
          product_id: "product-2", market_fetched_at: null },
      ],
    };
  }
  return { status: "not_recorded", chain: null, receipts_through: receiptsThrough, receipts: [] };
}

function wagers(week: number): Recap["wagers"] {
  if (week === 3) {
    const none = wagerSummary({
      bets: 0, settled: 0, unsettled: 0, wins: 0, losses: 0, staked: 0, settled_staked: 0, pnl: 0,
      roi_pct: null, bonus_funded: 0, clv_unavailable: 0,
    });
    return { total: none, markets: [], ledger_digest: "d".repeat(64) };
  }
  return {
    total: wagerSummary(),
    markets: [
      { market_type: "moneyline", summary: wagerSummary({ bets: 16, bonus_funded: 0 }), clv_kind: null,
        median_clv: null, share_positive_clv: null },
      { market_type: "same_game_parlay",
        summary: wagerSummary({ bets: 1, settled: 1, unsettled: 0, wins: 0, losses: 1, staked: 0.1, pnl: -0.1 }),
        clv_kind: null, median_clv: null, share_positive_clv: null },
    ],
    ledger_digest: "d".repeat(64),
  };
}

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
      margin_std: 13.5, actual_starter: "nflverse id when he threw a pass, else the top passer",
    },
    counts: { scheduled_games: 4, in_product: 3, not_in_product: 1, carried_rows: 1, outcome_ties: 1 },
    metrics: {
      win: { model: probability, market: { ...probability, brier: 0.1815 }, closer: { model: 1, market: 0, tie: 0 } },
      spread: { model: errors, market: { ...errors, mae: 3.1 }, closer: { model: 2, market: 1, tie: 0 } },
      total: { model: errors, market: errors, closer: { model: 1, market: 1, tie: 0 } },
      projected_score: { model: { ...errors, count: 4 }, market: errors, closer: { model: 1, market: 1, tie: 0 } },
    },
    games: [game("2026_02_AAA_BBB"), tied, notInProduct, totalMissing],
    starting_qbs: {
      slots: 8, scored: 4, from_capture: 2, not_recorded: 2, starter_unresolved: 0, not_in_product: 2,
      actual_missing: 0, actual_from_game_logs: 1,
      assumed: { count: 4, correct: 3, accuracy: 0.75 },
      nflverse: { count: 4, correct: 3, accuracy: 0.75 },
      carry_forward: { count: 4, correct: 3, accuracy: 0.75 },
      capture_digests: ["c".repeat(64)],
    },
    edges: edges(week), wagers: wagers(week),
  };
}

function loaded(items: Summary[]) {
  vi.mocked(useRecaps).mockReturnValue({ data: { items }, isLoading: false, error: null } as never);
  vi.mocked(useWeeklyRecap).mockImplementation(((season: string | null, week: number | null) => ({
    data: season && week ? recap(week) : undefined, isLoading: false, error: null,
  })) as never);
  vi.mocked(useSeasonRecap).mockReturnValue({ data: null, isLoading: false, error: null } as never);
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

  it("shows a production-chain week's issuance, recommendations and candidates without dollars", () => {
    window.history.replaceState(null, "", "#/recap?season=2026-2027&week=1");
    loaded([summary(1, "live", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const card = document.querySelector('[data-recap-section="edges"]') as HTMLElement;
    expect(card.querySelector('[data-edge-status="production_chain"]')).not.toBeNull();
    expect(within(card).getByText(/1680 rows: 698 candidates, 982 not candidates/)).toBeInTheDocument();
    expect(within(card).getByText(/698 unavailable/)).toBeInTheDocument();
    const moneyline = card.querySelector('[data-candidate-market="moneyline"]') as HTMLElement;
    expect(within(moneyline).getByText("104 kickoff conflict")).toBeInTheDocument();
    expect(within(moneyline).getByText("118 closed")).toBeInTheDocument();
    expect(within(moneyline).getByText("+0.064")).toBeInTheDocument();
    expect(within(moneyline).getByText("74.6%")).toBeInTheDocument();
    expect(within(moneyline).getByText("110-118-0")).toBeInTheDocument();
    const spread = card.querySelector('[data-candidate-market="spread"]') as HTMLElement;
    expect(within(spread).getByText("No candidate rows")).toHaveAttribute("data-weekly-status", "no_candidates");
    expect(within(card).getByText(/no dollars/)).toBeInTheDocument();
    expect(within(card).getByText("No weekly-cycle edge check before the last kickoff")).toBeInTheDocument();
  });

  it("says a withheld week published earlier and lists every edge check", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const card = document.querySelector('[data-recap-section="edges"]') as HTMLElement;
    expect(card.querySelector('[data-edge-status="withheld_unvalidated"]')).not.toBeNull();
    expect(within(card).getByText(/Earlier runs listed below published edges; their rows were not kept/)).toBeInTheDocument();
    const checks = within(card).getByRole("table", { name: /Every weekly-cycle edge check/ });
    expect(within(checks).getByText("published")).toBeInTheDocument();
    expect(within(checks).getByText("149 edges (top EV 27.1%)")).toBeInTheDocument();
    expect(within(checks).getByText(WITHHELD)).toBeInTheDocument();
  });

  it("states a week with no edge record and no wagers explicitly", () => {
    window.history.replaceState(null, "", "#/recap?season=2026-2027&week=3");
    loaded([summary(3, "live", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const edgesCard = document.querySelector('[data-recap-section="edges"]') as HTMLElement;
    expect(edgesCard.querySelector('[data-edge-status="not_recorded"]')).not.toBeNull();
    expect(within(edgesCard).getByText(/the edge report is a file that each run rewrites/)).toBeInTheDocument();
    const wagersCard = document.querySelector('[data-recap-section="wagers"]') as HTMLElement;
    expect(within(wagersCard).getByText("No recorded wagers on this week's games")).toBeInTheDocument();
  });

  it("totals the recorded wagers and states what the headline leaves out", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const card = document.querySelector('[data-recap-section="wagers"]') as HTMLElement;
    const tile = (label: string) => card.querySelector(`[data-wager-tile="${label}"]`) as HTMLElement;
    expect(within(tile("Bets")).getByText("17")).toBeInTheDocument();
    expect(within(tile("Won-lost-push")).getByText("10-6-0")).toBeInTheDocument();
    expect(within(tile("Staked")).getByText("$16.10")).toBeInTheDocument();
    expect(within(tile("P&L")).getByText("−$0.44")).toBeInTheDocument();
    expect(within(tile("ROI")).getByText("-2.9%")).toBeInTheDocument();
    expect(card.querySelector('[data-wager-note="unsettled"]')?.textContent).toMatch(/1 still open/);
    expect(card.querySelector('[data-wager-note="clv_unavailable"]')?.textContent).toMatch(/CLV unavailable for 17/);
    expect(card.querySelector('[data-wager-note="bonus_funded"]')?.textContent).toBe("1 bonus-funded");
    expect(card.querySelector('[data-wager-note="linked"]')?.textContent).toBe("None linked to a recommendation");
    const parlay = card.querySelector('[data-wager-market="same_game_parlay"]') as HTMLElement;
    expect(within(parlay).getByText("Same-game parlay")).toBeInTheDocument();
    expect(within(parlay).getByText("CLV unavailable")).toHaveAttribute("data-weekly-status", "clv_unavailable");
  });

  it("says when the season has no selected season recap", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    expect(useSeasonRecap).toHaveBeenLastCalledWith(SEASON);
    const card = document.querySelector('[data-recap-section="season_to_date"]') as HTMLElement;
    expect(within(card).getByText(`No season recap is selected for ${SEASON}`)).toBeInTheDocument();
    expect(within(card).getByText(/--season-only/)).toBeInTheDocument();
  });

  it("scores the starting quarterbacks and names every miss and unscored slot", () => {
    loaded([summary(2, "development", "2026-10-05T00:00:00Z")]);
    renderScreen();
    const section = document.querySelector('[data-recap-section="starting_qbs"]') as HTMLElement;
    expect(within(section).queryByTitle("Coming soon")).toBeNull();
    expect(within(section).getByText(/4 of 8 team slots scored · 2 from the captured sidecar/)).toBeInTheDocument();
    expect(within(section).getByText(/Who started: nflverse id when he threw a pass/)).toBeInTheDocument();
    const accuracy = within(section).getByRole("table", { name: "Starter predictor accuracy" });
    const assumed = within(accuracy).getByRole("row", { name: /Assumed by the forecast/ });
    expect(within(assumed).getByText("3 / 4")).toBeInTheDocument();
    expect(within(assumed).getByText("75.0%")).toBeInTheDocument();

    const misses = within(section).getByRole("region", { name: "Starter misses" });
    const wrongAssumption = within(misses).getByRole("row", { name: /^BBB/ });
    expect(within(wrongAssumption).getByText(/Wrong Rookie/)).toHaveAttribute("data-starter-pick", "wrong");
    const stale = within(misses).getByRole("row", { name: /^CCC/ });
    expect(within(stale).getByText(/Stale Listing/, { selector: "td:nth-child(3) span" }))
      .toHaveAttribute("data-starter-pick", "wrong");
    expect(within(stale).getByText(/from the captured sidecar of its run/)).toBeInTheDocument();
    expect(within(stale).getByText("top passer (nflverse value had no attempt)")).toBeInTheDocument();
    expect(within(misses).queryByRole("row", { name: /^AAA/ })).toBeNull();

    const notRecorded = section.querySelector('[data-starter-status="not_recorded"]') as HTMLElement;
    expect(within(notRecorded).getByText("GGG, HHH")).toBeInTheDocument();
    expect(within(notRecorded).getByText("Starting quarterbacks were not recorded for this forecast"))
      .toBeInTheDocument();
    const notInProduct = section.querySelector('[data-starter-status="not_in_product"]') as HTMLElement;
    expect(within(notInProduct).getByText("EEE, FFF")).toBeInTheDocument();
  });

  it("explains how to build a recap when none is selected", () => {
    loaded([]);
    renderScreen();
    expect(screen.getByText("No weekly recap has been built yet")).toBeInTheDocument();
    expect(useWeeklyRecap).toHaveBeenLastCalledWith(null, null);
  });
});
