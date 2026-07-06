type PillProps = {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
};

/**
 * Shared filter toggle button. Two states: active (accent) and
 * inactive (dim). Used across screens for cohort switchers, filter
 * groups, and mode toggles.
 *
 * Kept intentionally simple: state is caller-managed. If you need
 * groups with mutual exclusion or radio-like behavior, compose in
 * the caller — Pill itself does no coordination.
 */
export function Pill({ active, onClick, disabled = false, children }: PillProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      style={{
        background: active
          ? "var(--pos)"
          : "transparent",
        color: active
          ? "var(--bg)"
          : disabled
            ? "var(--ink-4)"
            : "var(--ink-2)",
        border: `1px solid ${
          active ? "var(--pos)" : "var(--line-soft)"
        }`,
        borderRadius: 4,
        padding: "4px 12px",
        fontSize: 11.5,
        fontFamily: "var(--f-sans)",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "background 90ms ease, color 90ms ease",
      }}
    >
      {children}
    </button>
  );
}
