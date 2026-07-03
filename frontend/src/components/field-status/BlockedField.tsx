type BlockedFieldProps = {
  blocker: string;
  roadmap: string;
  /** Optional custom placeholder. Defaults to em dash. */
  placeholder?: string;
};

/**
 * Renders a placeholder for a field marked `field_status: BlockedStatus`.
 * The field is not available — hovering shows the blocker slug and
 * roadmap reference so the user (or reviewer) knows why.
 */
export function BlockedField({
  blocker,
  roadmap,
  placeholder = "—",
}: BlockedFieldProps) {
  const tooltip = `Not available: ${blocker} (${roadmap})`;
  return (
    <span
      title={tooltip}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        color: "var(--ink-4)",
      }}
    >
      <span className="mono tnum">{placeholder}</span>
      <StatusBadge />
    </span>
  );
}

function StatusBadge() {
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
        border: "1px solid var(--warn-dim)",
        color: "var(--warn-dim)",
        fontSize: 9,
        fontWeight: 600,
        lineHeight: 1,
        cursor: "help",
      }}
    >
      !
    </span>
  );
}
