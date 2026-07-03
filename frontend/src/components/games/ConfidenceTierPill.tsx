type ConfidenceTierPillProps = {
  tier: string | null | undefined;
};

export function ConfidenceTierPill({ tier }: ConfidenceTierPillProps) {
  if (!tier) {
    return <span className="dim mono">—</span>;
  }

  const color =
    tier === "High"
      ? "var(--pos)"
      : tier === "Moderate"
        ? "var(--warn)"
        : "var(--ink-3)";

  return (
    <span
      className="mono upper"
      style={{
        fontSize: 10,
        color,
        padding: "2px 6px",
        border: `1px solid ${color}`,
        borderRadius: 3,
        display: "inline-block",
      }}
    >
      {tier}
    </span>
  );
}
