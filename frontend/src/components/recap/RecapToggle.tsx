/** A pressed-state toggle button shared by the recap page's filter rows. */
export function RecapToggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: string;
}) {
  return (
    <button type="button" aria-pressed={active} onClick={onClick} style={{
      padding: "7px 12px", border: "1px solid var(--line-soft)", borderRadius: 4,
      background: active ? "var(--bg-2)" : "transparent", color: active ? "var(--ink)" : "var(--ink-3)",
      fontFamily: "var(--f-sans)", fontSize: 10, cursor: "pointer",
    }}>{children}</button>
  );
}
