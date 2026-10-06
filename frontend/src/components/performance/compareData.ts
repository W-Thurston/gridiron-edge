import type { components } from "../../api/schema";

type SeasonRecap = components["schemas"]["SeasonRecapResponse"];
type WeekMetrics = components["schemas"]["RecapWeekMetrics"];

export type CompareModelGroup = "ours" | "market" | "external";

/**
 * Every model the comparison chart can draw. Each id has a fixed color, so a model keeps its color
 * whatever else is selected. A source declares only the models it has evidence for.
 */
export type CompareModelId =
  | "champion"
  | "challenger"
  | "baseline"
  | "opening_line"
  | "closing_line"
  | "external_a"
  | "external_b";

export type CompareModel = {
  id: CompareModelId;
  label: string;
  /** Where the values come from, shown beside the label. */
  source: string;
  group: CompareModelGroup;
};

/** One metric per period per model; null where the model has no value for that period. */
export type CompareData = {
  models: CompareModel[];
  periods: string[];
  values: Record<string, Partial<Record<CompareModelId, (number | null)[]>>>;
};

export type CompareMetric = {
  key: string;
  label: string;
  /** Column and tooltip name. */
  short: string;
  better: "lower" | "higher";
  format: (value: number) => string;
  /** Format for a difference; percentages are shown in points. */
  formatDelta: (value: number) => string;
  note?: string;
};

/** Either persisted comparison evidence, or the stated reason there is none. */
export type CompareSource =
  | { status: "available"; data: CompareData; metrics: CompareMetric[] }
  | { status: "unavailable"; reason: string; detail: string };

/**
 * Fixed series styles. Categorical hues are the dark steps of the dataviz reference palette in its
 * fixed order (validated against --bg-1); market lines are neutral greys and the opening line is dashed.
 */
export const MODEL_STYLES: Record<CompareModelId, { color: string; dash?: boolean }> = {
  champion: { color: "#3987e5" },
  challenger: { color: "#d95926" },
  baseline: { color: "#199e70" },
  external_a: { color: "#9085e9" },
  external_b: { color: "#d55181" },
  closing_line: { color: "var(--ink-3)" },
  opening_line: { color: "var(--ink-2)", dash: true },
};

export const GROUP_LABELS: Record<CompareModelGroup, string> = {
  ours: "Ours",
  market: "Market",
  external: "External",
};

export const SEASON_DEFAULT_MODELS: CompareModelId[] = ["champion", "closing_line"];
export const BACKTEST_DEFAULT_MODELS: CompareModelId[] = ["champion", "challenger", "closing_line"];

const fixedFormat = (digits: number) => (value: number) => value.toFixed(digits);

export const SEASON_METRICS: CompareMetric[] = [
  { key: "win_brier", label: "Moneyline Brier", short: "Brier", better: "lower", format: fixedFormat(3), formatDelta: fixedFormat(3) },
  { key: "spread_mae", label: "Spread MAE", short: "MAE", better: "lower", format: fixedFormat(2), formatDelta: fixedFormat(2) },
  { key: "total_mae", label: "Total MAE", short: "MAE", better: "lower", format: fixedFormat(2), formatDelta: fixedFormat(2) },
  { key: "score_mae", label: "Team score MAE", short: "MAE / team", better: "lower", format: fixedFormat(2), formatDelta: fixedFormat(2) },
];

const SEASON_PICKS: Record<string, (metrics: WeekMetrics) => [number | null | undefined, number | null | undefined]> = {
  win_brier: (m) => [m.win.model.brier, m.win.market.brier],
  spread_mae: (m) => [m.spread.model.mae, m.spread.market.mae],
  total_mae: (m) => [m.total.model.mae, m.total.market.mae],
  score_mae: (m) => [m.projected_score.model.mae, m.projected_score.market.mae],
};

export type SeasonGames = "all" | "live";
export type SeasonView = "week" | "to_date";

/** A week's metrics in the chosen games filter and view; null for an unbuilt week or one with no live games. */
export function seasonWeekMetrics(
  week: SeasonRecap["weeks"][number],
  games: SeasonGames,
  view: SeasonView,
): WeekMetrics | null {
  if (week.status === "not_built") return null;
  if (view === "week" && games === "live" && week.live_games === 0) return null;
  const metrics = view === "week"
    ? (games === "all" ? week.metrics : week.live_metrics)
    : (games === "all" ? week.to_date : week.live_to_date);
  return metrics ?? null;
}

/** The selected season recap as comparison data: the published model and the closing line, week by week. */
export function seasonComparison(recap: SeasonRecap, games: SeasonGames, view: SeasonView): CompareData {
  const weeks = recap.weeks;
  const values: CompareData["values"] = {};
  for (const [key, pick] of Object.entries(SEASON_PICKS)) {
    const pairs = weeks.map((week) => {
      const metrics = seasonWeekMetrics(week, games, view);
      return metrics ? pick(metrics) : [null, null];
    });
    values[key] = {
      champion: pairs.map(([model]) => model ?? null),
      closing_line: pairs.map(([, market]) => market ?? null),
    };
  }
  return {
    models: [
      { id: "champion", label: "Champion", source: "Published forecasts", group: "ours" },
      { id: "closing_line", label: "Closing line", source: "Closing spread and total", group: "market" },
    ],
    periods: weeks.map((week) => `W${week.week}`),
    values,
  };
}

/** The backtest has no persisted per-season comparison yet; say so rather than derive one in the browser. */
export const BACKTEST_COMPARISON: CompareSource = {
  status: "unavailable",
  reason: "Per-season comparison is not persisted yet",
  detail:
    "The historical report stores per-game evidence and a cumulative series, but no season-by-season rollup for " +
    "the champion, challengers or the closing line. Rolling one up in the browser would reconstruct metrics the " +
    "report never published, so this chart waits for the report to persist it.",
};
