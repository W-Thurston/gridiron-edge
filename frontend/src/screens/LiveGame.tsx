import { BlockedScreen } from "../components/blocked/BlockedScreen";

export function LiveGame() {
  return (
    <BlockedScreen
      title="Live Game"
      description="Real-time win probability updates during game play, live odds tracking, drive-by-drive line movement, and hedge calculator for in-game bet management."
      blocker="live_state_ingest"
      roadmap="W10"
      requirements={[
        "Live game state ingest (score, clock, down/distance, possession)",
        "Live win probability model",
        "Live odds ingest (per-book, per-market)",
        "WebSocket API for real-time frontend updates",
      ]}
    />
  );
}
