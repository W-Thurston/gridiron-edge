import type { components } from "../../api/schema";

export type RecapFamily = "win" | "spread" | "total" | "projected_score";
type ForecastStatus = components["schemas"]["ForecastStatus"];
type ForecastMarketStatus = components["schemas"]["ForecastMarketStatus"];
type ClosingLineStatus = components["schemas"]["ClosingLineStatus"];
type ForecastRole = components["schemas"]["ForecastRole"];

const FAMILY_LABELS: Record<RecapFamily, string> = {
  win: "Moneyline",
  spread: "Spread",
  total: "Total",
  projected_score: "Team score",
};

export function recapFamilyLabel(family: RecapFamily): string {
  return FAMILY_LABELS[family];
}

/** Full sentence for a family's forecast status (titles, aria labels, value cells). */
export function recapForecastStatusMessage(
  family: RecapFamily,
  status: ForecastStatus,
): string {
  switch (status) {
    case "available":
      return `${FAMILY_LABELS[family]} forecast available`;
    case "forecast_missing":
      return `${FAMILY_LABELS[family]} forecast missing from the selected product`;
    case "not_in_product":
      return "No forecast in the selected product";
    case "outcome_tie":
      return "Tied game: no home-win outcome to score";
  }
}

/** Short label for the same status where a full sentence would repeat across a row. */
export function recapForecastStatusShort(status: ForecastStatus): string {
  switch (status) {
    case "available":
      return "Available";
    case "forecast_missing":
      return "Forecast missing";
    case "not_in_product":
      return "Not in product";
    case "outcome_tie":
      return "Tied game";
  }
}

export function recapMarketAtForecastMessage(status: ForecastMarketStatus): string {
  switch (status) {
    case "available":
      return "Market at forecast time recorded";
    case "not_recorded":
      return "Market at forecast time not recorded";
    case "not_in_product":
      return "No forecast in the selected product";
  }
}

export function recapClosingLineMessage(status: ClosingLineStatus): string {
  switch (status) {
    case "available":
      return "Closing line available";
    case "closing_line_missing":
      return "No closing line";
  }
}

export function recapRoleLabel(role: ForecastRole | null): string {
  switch (role) {
    case "live":
      return "Live forecast";
    case "development":
      return "Development forecast: regenerated after the fact, not issued before kickoff";
    case "backfilled":
      return "Backfilled forecast";
    case null:
      return "Forecast role not recorded";
  }
}

type StarterStatus = components["schemas"]["StarterStatus"];
type ActualStarterSource = components["schemas"]["ActualStarterSource"];
type StarterRecordSource = components["schemas"]["StarterRecordSource"];

/** Full sentence for why a starter slot is or is not scored. */
export function recapStarterStatusMessage(status: StarterStatus): string {
  switch (status) {
    case "available":
      return "Starter assumptions scored";
    case "not_recorded":
      return "Starting quarterbacks were not recorded for this forecast";
    case "starter_unresolved":
      return "The forecast resolved no starter for this side";
    case "not_in_product":
      return "No forecast in the selected product";
    case "actual_missing":
      return "No pass attempt for the team in the game logs, so the actual starter is unknown";
  }
}

/** Short label for the same status, used where teams are grouped by status. */
export function recapStarterStatusShort(status: StarterStatus): string {
  switch (status) {
    case "available":
      return "Scored";
    case "not_recorded":
      return "Not recorded";
    case "starter_unresolved":
      return "Starter unresolved";
    case "not_in_product":
      return "Not in product";
    case "actual_missing":
      return "Actual starter unknown";
  }
}

export function recapActualSourceLabel(source: ActualStarterSource): string {
  switch (source) {
    case "nflverse_schedule":
      return "nflverse";
    case "game_log_top_passer":
      return "top passer (nflverse value had no attempt)";
  }
}

export function recapRecordSourceLabel(source: StarterRecordSource): string {
  switch (source) {
    case "product_record":
      return "recorded in the product";
    case "sidecar_capture":
      return "from the captured sidecar of its run";
  }
}

type EdgeEvidenceStatus = components["schemas"]["EdgeEvidenceStatus"];
type MarketCloseoutStatus = components["schemas"]["MarketCloseoutStatus"];
type CandidateRecommendationState = components["schemas"]["CandidateRecommendationState"];
type MarketClvKind = components["schemas"]["MarketClvKind"];

/** Full sentence for what a week's persisted state says about the edges it published. */
export function recapEdgeStatusMessage(status: EdgeEvidenceStatus): string {
  switch (status) {
    case "production_chain":
      return "Recorded by the production chain: the candidate issuance of the selected product";
    case "withheld_unvalidated":
      return "Withheld: model edges are not validated, so the last edge check before kickoff published none";
    case "not_recorded":
      return "Not recorded: the edge report is a file that each run rewrites, so it is not evidence of what was published";
  }
}

export function recapEdgeStatusShort(status: EdgeEvidenceStatus): string {
  switch (status) {
    case "production_chain":
      return "Production chain";
    case "withheld_unvalidated":
      return "Withheld";
    case "not_recorded":
      return "Not recorded";
  }
}

/** Plain words for a market-closeout status (snake case as words). */
export function recapCloseoutStatusLabel(status: MarketCloseoutStatus): string {
  return status === "available" ? "closed" : status.replaceAll("_", " ");
}

export function recapRecommendationStateLabel(state: CandidateRecommendationState): string {
  return state === "not_evaluated" ? "not evaluated" : state;
}

/** The unit a CLV value is stated in. */
export function recapClvKindLabel(kind: MarketClvKind): string {
  switch (kind) {
    case "moneyline_price":
      return "relative to the bet's implied probability";
    case "spread_points":
      return "points";
    case "total_points":
      return "points";
  }
}
