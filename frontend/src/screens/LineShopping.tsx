import { BlockedScreen } from "../components/blocked/BlockedScreen";

export function LineShopping() {
  return (
    <BlockedScreen
      title="Line Shopping"
      description="Compare odds across multiple sportsbooks to find the best price on every market. Cross-book edge detection, arbitrage opportunities, and middle-market discovery."
      blocker="multi_book_ingest"
      roadmap="multi-book markets"
      requirements={[
        "Multi-book sportsbook ingest pipeline; current game markets use the nflverse schedule source",
        "Odds source decision — likely The Odds API for ~15 books coverage",
        "market/line_shopping.py: best_price, price_comparison_table, detect_arbitrage",
      ]}
    />
  );
}
