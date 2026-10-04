import {
  formatMarketAge,
  marketAge,
  type MarketFreshnessContext,
} from "./marketAge";

/**
 * When the quotes were fetched and how old that is against the limit. A stale
 * snapshot is stated, never hidden: the server withholds edges and price guidance
 * for it, and this says why.
 */
export function MarketAsOf({
  fetchedAt,
  freshness,
}: {
  fetchedAt: readonly string[] | string | null | undefined;
  freshness: MarketFreshnessContext;
}) {
  const age = marketAge(fetchedAt, freshness);

  if (age.state === "unknown") {
    return (
      <div
        className="mono dim"
        role="note"
        data-market-freshness="unknown"
        style={{ fontSize: 11 }}
      >
        Market snapshot time unavailable
      </div>
    );
  }

  const fetched = new Date(age.fetchedAt as string).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
  const stale = age.state === "stale";
  return (
    <div
      className="mono dim"
      role="note"
      data-market-freshness={age.state}
      style={{ fontSize: 11, color: stale ? "var(--warn)" : undefined }}
    >
      Market as of {fetched} · {formatMarketAge(age.ageMs as number)} old
      {stale
        ? ` · older than the ${age.limitHours} h limit; edges and price guidance are withheld`
        : ""}
    </div>
  );
}
