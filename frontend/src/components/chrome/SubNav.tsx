type SubNavProps = {
  /** Left-slot content — typically breadcrumbs or a title. */
  left?: React.ReactNode;
  /** Right-slot content — typically filters, tabs, or actions. */
  right?: React.ReactNode;
};

export function SubNav({ left, right }: SubNavProps) {
  return (
    <div
      style={{
        height: 44,
        background: "var(--bg)",
        borderBottom: "1px solid var(--line-soft)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>{left}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {right}
      </div>
    </div>
  );
}
