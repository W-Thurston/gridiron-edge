import type { components } from "../../api/schema";
type RecentResult = components["schemas"]["RecentResult"];

type RecentResultsStripProps = {
  results: RecentResult[] | null | undefined;
};

export function RecentResultsStrip({ results }: RecentResultsStripProps) {
  if (!results || results.length === 0) {
    return <span className="dim mono">—</span>;
  }

  return (
    <div style={{ display: "flex", gap: 6 }}>
      {results.map((result, i) => (
        <ResultPill key={i} result={result} />
      ))}
    </div>
  );
}

function ResultPill({ result }: { result: RecentResult }) {
  const outcome = result.result ?? "?";
  const color =
    outcome === "W"
      ? "var(--pos)"
      : outcome === "L"
        ? "var(--neg)"
        : "var(--ink-3)";

  const tooltip = [
    result.date && `Week ${result.week} (${result.date})`,
    result.opponent && `${result.is_home ? "vs" : "@"} ${result.opponent}`,
    result.score_for != null &&
      result.score_against != null &&
      `${result.score_for}-${result.score_against}`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <span
      title={tooltip}
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 22,
        height: 22,
        borderRadius: 3,
        color: color,
        border: `1px solid ${color}`,
        fontSize: 11,
        fontWeight: 600,
        cursor: "help",
      }}
    >
      {outcome}
    </span>
  );
}
