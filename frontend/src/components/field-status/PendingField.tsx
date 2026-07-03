type PendingFieldProps = {
  /** Optional custom placeholder. Defaults to em dash. */
  placeholder?: string;
};

/**
 * Renders a placeholder for a field marked `field_status: "pending"`.
 * Backend work is planned but not done — the field will populate in a
 * future backend workstream (typically W8 Tier 3).
 */
export function PendingField({ placeholder = "—" }: PendingFieldProps) {
  return (
    <span
      title="Coming soon"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        color: "var(--ink-4)",
      }}
    >
      <span className="mono tnum">{placeholder}</span>
      <StatusBadge kind="pending" />
    </span>
  );
}

function StatusBadge({ kind }: { kind: "pending" | "blocked" }) {
  const glyph = kind === "pending" ? "i" : "!";
  const color = kind === "pending" ? "var(--info)" : "var(--warn-dim)";
  return (
    <span
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 12,
        height: 12,
        borderRadius: "50%",
        border: `1px solid ${color}`,
        color,
        fontSize: 9,
        fontWeight: 600,
        lineHeight: 1,
        cursor: "help",
      }}
    >
      {glyph}
    </span>
  );
}
