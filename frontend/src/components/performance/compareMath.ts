import type { CompareData, CompareMetric, CompareModel } from "./compareData";

/** Runs of consecutive periods with a value; a line breaks wherever a model has none. */
export function segments(series: (number | null)[]): [number, number][][] {
  const out: [number, number][][] = [];
  let current: [number, number][] = [];
  series.forEach((v, i) => {
    if (v == null) {
      if (current.length) out.push(current);
      current = [];
    } else {
      current.push([i, v]);
    }
  });
  if (current.length) out.push(current);
  return out;
}

/** Every visible model's value at one period, best first. */
export function tooltipRows(data: CompareData, metric: CompareMetric, visible: CompareModel[], index: number) {
  const values = data.values[metric.key] ?? {};
  return visible
    .map((model) => ({ model, value: values[model.id]?.[index] ?? null }))
    .filter((row): row is { model: CompareModel; value: number } => row.value != null)
    .sort((a, b) => (metric.better === "lower" ? a.value - b.value : b.value - a.value));
}

type Summary = {
  model: CompareModel;
  average: number | null;
  /** Mean of (model - champion) over the periods both cover. */
  delta: number | null;
  best: number;
  covered: number;
  first: number;
  last: number;
};

/** Per-model averages, difference from the champion, periods won and coverage; best average first. */
export function summarize(data: CompareData, metric: CompareMetric, visible: CompareModel[]): Summary[] {
  const values = data.values[metric.key] ?? {};
  const champion = values.champion ?? [];
  const isBetter = (a: number, b: number) => (metric.better === "lower" ? a < b : a > b);
  const rows = visible.map((model) => {
    const series = values[model.id] ?? [];
    const present = series.flatMap((v, i) => (v == null ? [] : [[i, v] as const]));
    const average = present.length ? present.reduce((sum, [, v]) => sum + v, 0) / present.length : null;
    const shared = present.flatMap(([i, v]) => (champion[i] == null ? [] : [v - (champion[i] as number)]));
    const delta = model.id === "champion" || shared.length === 0
      ? null
      : shared.reduce((sum, d) => sum + d, 0) / shared.length;
    const best = present.filter(([i, v]) =>
      visible.every((other) => {
        const ov = values[other.id]?.[i];
        return other.id === model.id || ov == null || !isBetter(ov, v);
      }),
    ).length;
    return {
      model, average, delta, best, covered: present.length,
      first: present.length ? present[0][0] : -1,
      last: present.length ? present[present.length - 1][0] : -1,
    };
  });
  return rows.sort((a, b) => {
    if (a.average == null) return 1;
    if (b.average == null) return -1;
    return metric.better === "lower" ? a.average - b.average : b.average - a.average;
  });
}
