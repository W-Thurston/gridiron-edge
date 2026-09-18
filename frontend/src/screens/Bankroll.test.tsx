import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Bankroll } from "./Bankroll";
import { usePortfolioBets, usePortfolioCurve, usePortfolioSplits, usePortfolioSummary, usePortfolioTransactions } from "../api/hooks";

vi.mock("../api/hooks", () => ({ usePortfolioBets: vi.fn(), usePortfolioCurve: vi.fn(), usePortfolioSplits: vi.fn(), usePortfolioSummary: vi.fn(), usePortfolioTransactions: vi.fn() }));
vi.mock("../context/AppStateContext", () => ({ useAppState: () => ({ state: { oddsFormat: "american" } }) }));
vi.mock("../components/portfolio/BalanceCurve", () => ({ BalanceCurve: ({ points }: { points: unknown[] }) => <div data-testid="balance-curve">{points.length}</div> }));
function queryResult<T>(data: T): unknown { return { data, error: null, isLoading: false, refetch: vi.fn() }; }

const bets = [
  { bet_id: "b1", game_id: "2026_01_KC_LAC", description: "Kansas City Chiefs", placed_at: "2026-09-07T18:00:00Z", market_type: "moneyline", side: "home", odds: -150, stake: 1, book: "draftkings", funding_type: "cash", status: "won", pnl: 0.67 },
  { bet_id: "b2", source_bet_id: "DK-2", game_id: null, description: "19-pick SGP parlay", placed_at: "2026-09-12T14:00:20Z", market_type: "same_game_parlay", side: null, odds: 145273632, stake: 0.1, book: "draftkings", funding_type: "bonus", status: "lost", pnl: -0.1 },
  ...Array.from({ length: 19 }, (_, index) => ({ bet_id: `open-${index}`, game_id: `2026_02_GAME_${index}`, description: `Open wager ${index}`, placed_at: "2026-09-16T18:00:00Z", market_type: "moneyline", side: "home", odds: -110, stake: 1, book: "betmgm", funding_type: "cash", status: "open", pnl: null })),
];
const transactions = Array.from({ length: 21 }, (_, index) => ({ txn_id: `t${index}`, timestamp: `2026-09-${String(index + 1).padStart(2, "0")}T18:00:00Z`, txn_type: index === 0 ? "deposit" : "bet_placed", amount: index === 0 ? 60.7 : 1, balance_after: 60.7 - Math.max(index, 0) }));

describe("Bankroll", () => {
  beforeEach(() => {
    vi.mocked(usePortfolioSummary).mockReturnValue(queryResult({ bankroll: 41.96, wins: 1, losses: 1, pushes: 0, win_pct: 0.5, total_bets: bets.length, open_bets: 19, total_pnl: 0.57, roi_pct: 51.8, total_staked: 1.1, mean_clv: null, current_streak: "L1" }) as ReturnType<typeof usePortfolioSummary>);
    vi.mocked(usePortfolioCurve).mockReturnValue(queryResult({ items: [{ timestamp: "2026-09-07", bankroll: 60.7 }, { timestamp: "2026-09-17", bankroll: 41.96 }], total: 2, period: null }) as ReturnType<typeof usePortfolioCurve>);
    vi.mocked(usePortfolioBets).mockReturnValue(queryResult({ total: bets.length, items: bets }) as ReturnType<typeof usePortfolioBets>);
    vi.mocked(usePortfolioTransactions).mockReturnValue(queryResult({ total: transactions.length, items: transactions }) as ReturnType<typeof usePortfolioTransactions>);
    vi.mocked(usePortfolioSplits).mockReturnValue(queryResult({ items: [], total: 0, dimension: "market_type" }) as ReturnType<typeof usePortfolioSplits>);
  });

  it("shows available-balance context and bounded scrolling regions", () => {
    render(<Bankroll />);
    expect(screen.getByText("Available Bankroll")).toBeInTheDocument();
    expect(screen.getByText("Since start:")).toBeInTheDocument();
    expect(screen.getByText("-$18.74 (-30.9%)")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: `Bets, ${bets.length} in view` })).toHaveStyle({ overflowY: "auto" });
    expect(screen.getByRole("region", { name: `Transactions, ${transactions.length} total` })).toHaveStyle({ overflowY: "auto" });
  });

  it("filters the Record card by market", () => {
    render(<Bankroll />);
    fireEvent.change(screen.getByLabelText("Record market"), { target: { value: "same_game_parlay" } });
    expect(screen.getByText("0-1")).toBeInTheDocument();
    expect(screen.getByText("Bets in View").parentElement).toHaveTextContent("1");
  });

  it("combines bet filters and clears them", () => {
    render(<Bankroll />);
    fireEvent.change(screen.getByLabelText("Bet status"), { target: { value: "lost" } });
    fireEvent.change(screen.getByLabelText("Bet market"), { target: { value: "same_game_parlay" } });
    const region = screen.getByRole("region", { name: "Bets, 1 in view" });
    expect(within(region).getByText("19-pick SGP parlay")).toBeInTheDocument();
    expect(within(region).queryByText("2026_01_KC_LAC")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByRole("region", { name: `Bets, ${bets.length} in view` })).toBeInTheDocument();
  });

  it("filters by funding, sportsbook, and search", () => {
    render(<Bankroll />);
    fireEvent.click(screen.getByRole("button", { name: "Filters" }));
    fireEvent.change(screen.getByLabelText("Bet funding"), { target: { value: "bonus" } });
    fireEvent.change(screen.getByLabelText("Bet sportsbook"), { target: { value: "draftkings" } });
    fireEvent.change(screen.getByLabelText("Search bets"), { target: { value: "SGP" } });
    expect(screen.getByRole("region", { name: "Bets, 1 in view" })).toHaveTextContent("19-pick SGP parlay");
  });

  it("requests funding splits after selecting the Funding tab", () => {
    render(<Bankroll />);
    expect(usePortfolioSplits).toHaveBeenLastCalledWith("market_type");
    fireEvent.click(screen.getByRole("button", { name: "Funding" }));
    expect(usePortfolioSplits).toHaveBeenLastCalledWith("funding_type");
  });
});
