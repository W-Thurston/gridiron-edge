import { BlockedScreen } from "../components/blocked/BlockedScreen";

export function ExplainPage() {
  return (
    <BlockedScreen
      title="Explain"
      description='"Why does the model give this game 71%?" View feature contribution waterfall, historical comparable games, and what-if scenarios that show how predictions change under different conditions.'
      blocker="scenario_engine"
      roadmap="W4.5"
      requirements={[
        "Feature attribution / per-factor prediction decomposition",
        "Comparable game retrieval (nearest-neighbor over historical games)",
        "Scenario engine for what-if propagation (What if Mahomes is out?)",
        "Injury data source decision (blocks W4.5)",
      ]}
    />
  );
}
