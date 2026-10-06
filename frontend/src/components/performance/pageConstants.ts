import type { RecapFamily } from "../field-status/recapStatus";

export type Scope = "week" | "season" | "backtest";

/** The scope panel the scope tabs control. */
export const SCOPE_PANEL_ID = "performance-scope-panel";

/** Forecast families in the order the Week scope shows them. */
export const FAMILIES: RecapFamily[] = ["win", "spread", "total", "projected_score"];
