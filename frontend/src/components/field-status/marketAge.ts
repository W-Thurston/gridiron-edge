/** Market snapshot age against the server's freshness limit (the server enforces it). */

export type MarketFreshnessContext = {
  evaluated_at: string;
  max_age_seconds: number;
};

export type MarketAge = {
  state: "fresh" | "stale" | "unknown";
  fetchedAt: string | null;
  ageMs: number | null;
  limitHours: number;
};

export function latestTimestamp(
  values: readonly string[] | string | null | undefined,
): string | null {
  const list = typeof values === "string" ? [values] : (values ?? []);
  const valid = list.filter((value) => !Number.isNaN(new Date(value).getTime()));
  if (valid.length === 0) return null;
  return valid.reduce((latest, value) =>
    new Date(value).getTime() > new Date(latest).getTime() ? value : latest,
  );
}

export function marketAge(
  fetchedAt: readonly string[] | string | null | undefined,
  freshness: MarketFreshnessContext,
): MarketAge {
  const latest = latestTimestamp(fetchedAt);
  const limitHours = freshness.max_age_seconds / 3600;
  const evaluated = new Date(freshness.evaluated_at).getTime();
  if (latest === null || Number.isNaN(evaluated)) {
    return { state: "unknown", fetchedAt: latest, ageMs: null, limitHours };
  }
  const ageMs = Math.max(evaluated - new Date(latest).getTime(), 0);
  return {
    state: ageMs > freshness.max_age_seconds * 1000 ? "stale" : "fresh",
    fetchedAt: latest,
    ageMs,
    limitHours,
  };
}

export function formatMarketAge(ageMs: number): string {
  const minutes = ageMs / 60000;
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours.toFixed(1)} h`;
  return `${(hours / 24).toFixed(1)} days`;
}
