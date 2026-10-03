export type WeeklyComponent =
  | "win"
  | "spread"
  | "total"
  | "projected_score";

const STATUS_MESSAGES: Record<WeeklyComponent, Record<string, string>> = {
  win: {
    available: "Win prediction available",
    policy_unavailable: "Win prediction unavailable under the weekly policy",
    forecast_missing: "Win forecast missing",
    forecast_ambiguous: "Win forecast selection ambiguous",
  },
  spread: {
    available: "Spread available",
    win_unavailable: "Spread unavailable because Win is unavailable",
    calibration_unavailable: "Spread calibration unavailable",
  },
  total: {
    available: "Total available",
    uncertainty_unavailable: "Total available; uncertainty unavailable",
    policy_unavailable: "Total unavailable under the weekly policy",
    forecast_missing: "Total forecast missing",
    forecast_ambiguous: "Total forecast selection ambiguous",
  },
  projected_score: {
    available: "Projected score available",
    spread_unavailable:
      "Projected score unavailable because Spread is unavailable",
    total_unavailable:
      "Projected score unavailable because Total is unavailable",
    spread_and_total_unavailable:
      "Projected score unavailable because Spread and Total are unavailable",
  },
};

export function weeklyComponentStatusMessage(
  component: WeeklyComponent,
  status: string,
): string {
  return STATUS_MESSAGES[component][status] ??
    `${component.replace("_", " ")} status: ${status}`;
}

export function isWeeklyComponentUsable(
  component: WeeklyComponent,
  status: string,
): boolean {
  if (component === "total") {
    return status === "available" || status === "uncertainty_unavailable";
  }
  return status === "available";
}

const STARTING_QB_STATUS_MESSAGES: Record<string, string> = {
  available: "Starting quarterbacks recorded",
  starter_unresolved:
    "Starting quarterback unresolved for at least one team",
  sidecar_unavailable:
    "Starting quarterbacks unavailable: no quarterback inputs were recorded for this forecast",
  not_recorded:
    "Starting quarterbacks were not recorded for this forecast",
};

export function startingQbStatusMessage(status: string): string {
  return STARTING_QB_STATUS_MESSAGES[status] ??
    `Starting quarterback status: ${status}`;
}

const STARTING_QB_SOURCE_LABELS: Record<string, string> = {
  depth_chart_report: "Depth chart and injury report",
  nflverse_schedule: "nflverse schedule",
  carried_forward: "Carried forward from the last known starter",
};

export function startingQbSourceLabel(source: string): string {
  return STARTING_QB_SOURCE_LABELS[source] ?? source;
}
