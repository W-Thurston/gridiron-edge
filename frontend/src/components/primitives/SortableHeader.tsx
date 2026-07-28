type SortDirection = "asc" | "desc";

type SortableHeaderProps = {
  label: string;
  active: boolean;
  direction: SortDirection;
  onClick: () => void;
  align?: "left" | "right";
};

export function SortableHeader({
  label,
  active,
  direction,
  onClick,
  align = "left",
}: SortableHeaderProps) {
  const ariaSort = active
    ? direction === "asc"
      ? "ascending"
      : "descending"
    : "none";

  const glyph = active
    ? direction === "asc"
      ? "↑"
      : "↓"
    : "↕";

  return (
    <th
      scope="col"
      aria-sort={ariaSort}
      style={{
        padding: align === "right"
          ? "8px 12px 8px 0"
          : "8px 12px 8px 0",
        textAlign: align,
      }}
    >
      <button
        type="button"
        onClick={onClick}
        aria-label={`Sort by ${label}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: align === "right" ? "flex-end" : "flex-start",
          gap: 4,
          width: "100%",
          padding: 0,
          border: 0,
          background: "transparent",
          color: active ? "var(--ink-2)" : "var(--ink-3)",
          font: "inherit",
          textAlign: align,
          cursor: "pointer",
        }}
      >
        <span>{label}</span>
        <span
          aria-hidden="true"
          style={{
            color: active ? "var(--pos)" : "var(--ink-4)",
            fontSize: 10,
          }}
        >
          {glyph}
        </span>
      </button>
    </th>
  );
}
