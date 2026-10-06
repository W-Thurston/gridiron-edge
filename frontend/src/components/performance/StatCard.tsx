import type { ReactNode } from "react";

/** One headline number with an optional visual and a mono sub-line. */
export function StatCard({
  label,
  value,
  sub,
  tone,
  children,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "pos" | "dim";
  children?: ReactNode;
}) {
  return (
    <article className="hm-card pf-stat" data-stat={label}>
      <span className="pf-stat-label">{label}</span>
      <span className={`pf-stat-value${tone ? ` pf-tone-${tone}` : ""}`}>{value}</span>
      {children}
      {sub && <span className="pf-stat-sub">{sub}</span>}
    </article>
  );
}

/** Wins, pushes and losses as one proportional bar; decorative, the numbers are in the sub-line. */
export function RecordBar({ wins, losses, pushes = 0 }: { wins: number; losses: number; pushes?: number }) {
  if (wins + losses + pushes === 0) return null;
  return (
    <div className="pf-split-bar" aria-hidden="true" style={{ marginTop: 4 }}>
      <i style={{ flex: wins, background: "var(--pos)" }} />
      {pushes > 0 && <i style={{ flex: pushes, background: "var(--line)" }} />}
      <i style={{ flex: losses, background: "var(--neg-dim)" }} />
    </div>
  );
}
