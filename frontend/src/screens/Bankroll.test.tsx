import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Bankroll } from "./Bankroll";
import {
  usePortfolioBets,
  usePortfolioCurve,
  usePortfolioSplits,
  usePortfolioSummary,
  usePortfolioTransactions,
} from "../api/hooks";

vi.mock("../api/hooks", () => ({
  usePortfolioBets: vi.fn(),
  usePortfolioCurve: vi.fn(),
  usePortfolioSplits: vi.fn(),
  usePortfolioSummary: vi.fn(),
  usePortfolioTransactions: vi.fn(),
}));

vi.mock("../context/AppStateContext", () => ({
  useAppState: () => ({ state: { oddsFormat: "american" } }),
}));

vi.mock("../components/portfolio/BalanceCurve", () => ({
  BalanceCurve: () => <div data-testid="balance-curve" />,
}));

function queryResult<T>(data: T): unknown {
  return {
    data,
    error: null,
    isLoading: false,
    refetch: vi.fn(),
  };
}

describe("Bankroll", () => {
  beforeEach(() => {
    vi.mocked(usePortfolioSummary).mockReturnValue(queryResult({
      bankroll: 41.96,
      wins: 11,
      losses: 9,
      pushes: 0,
      win_pct: 0.55,
      total_bets: 40,
      open_bets: 20,
      total_pnl: -18.74,
      roi_pct: -54.3,
      total_staked: 34.5,
      mean_clv: null,
      current_streak: "L1",
    }) as ReturnType<typeof usePortfolioSummary>);
    vi.mocked(usePortfolioCurve).mockReturnValue(queryResult({ items: [], total: 0, period: null }) as ReturnType<typeof usePortfolioCurve>);
    vi.mocked(usePortfolioBets).mockReturnValue(queryResult({
      total: 2,
      items: [
        { bet_id: "b1", game_id: "2026_01_KC_LAC", description: "Kansas City Chiefs", placed_at: "2026-09-07T18:00:00Z", market_type: "moneyline", side: "home", odds: -150, stake: 1, funding_type: "cash", status: "won", pnl: 0.67 },
        { bet_id: "b2", source_bet_id: "DK-2", game_id: null, description: "19-pick SGP parlay", placed_at: "2026-09-12T14:00:20Z", market_type: "same_game_parlay", side: null, odds: 145273632, stake: 0.1, funding_type: "bonus", status: "lost", pnl: -0.1 },
      ],
    }) as ReturnType<typeof usePortfolioBets>);
    vi.mocked(usePortfolioTransactions).mockReturnValue(queryResult({
      total: 1,
      items: [{ txn_id: "t1", timestamp: "2026-09-16T18:00:00Z", txn_type: "bet_placed", amount: 1, balance_after: 41.96 }],
    }) as ReturnType<typeof usePortfolioTransactions>);
    vi.mocked(usePortfolioSplits).mockReturnValue(queryResult({ items: [], total: 0, dimension: "market_type" }) as ReturnType<typeof usePortfolioSplits>);
  });

  it("renders historical bets and source balances", () => {
    render(<Bankroll />);

    expect(screen.getByText("Available Bankroll")).toBeInTheDocument();
    expect(screen.getAllByText("$41.96").length).toBeGreaterThan(0);
    expect(screen.getByText("2026_01_KC_LAC")).toBeInTheDocument();
    expect(screen.getByText("19-pick SGP parlay")).toBeInTheDocument();
    expect(screen.getByText("Same Game Parlay")).toBeInTheDocument();
    expect(screen.getByText("Bonus")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Bet" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Balance" })).toBeInTheDocument();
  });

  it("requests funding splits after selecting the Funding tab", () => {
    render(<Bankroll />);

    expect(usePortfolioSplits).toHaveBeenLastCalledWith("market_type");
    fireEvent.click(screen.getByRole("button", { name: "Funding" }));
    expect(usePortfolioSplits).toHaveBeenLastCalledWith("funding_type");
    expect(screen.getByRole("button", { name: "Book" })).toBeInTheDocument();
  });
});
