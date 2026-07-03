import { BlockedScreen } from "../components/blocked/BlockedScreen";

export function NewsWire() {
  return (
    <BlockedScreen
      title="News Wire"
      description="Injury reports, roster moves, coaching decisions, and market-moving news filtered by game and impact severity. Alerts when news materially affects your open bets."
      blocker="news_ingest"
      roadmap="deferred"
      requirements={[
        "News data source decision (ESPN API, Rotowire, RotoWorld, or manual)",
        "Ingest pipeline for real-time news feed",
        "News → game_id impact classification",
        "Alert routing to open bets in the portfolio",
      ]}
    />
  );
}
