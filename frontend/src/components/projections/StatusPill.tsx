type StatusPillProps = {
  clinched: boolean | null | undefined;
  eliminated: boolean | null | undefined;
};

/**
 * Renders a small status pill for a team's playoff position.
 * - Clinched → green "CLINCHED".
 * - Eliminated → red "ELIMINATED".
 * - Neither → nothing (returns null).
 * - Both null → nothing (state is pending per API field_status).
 */
export function StatusPill({ clinched, eliminated }: StatusPillProps) {
  if (clinched === true) {
    return (
      <span
        className="mono upper"
        style={{
          fontSize: 9,
          color: "var(--pos)",
          padding: "2px 5px",
          border: "1px solid var(--pos)",
          borderRadius: 3,
        }}
      >
        Clinched
      </span>
    );
  }
  if (eliminated === true) {
    return (
      <span
        className="mono upper"
        style={{
          fontSize: 9,
          color: "var(--neg)",
          padding: "2px 5px",
          border: "1px solid var(--neg)",
          borderRadius: 3,
        }}
      >
        Elim.
      </span>
    );
  }
  return null;
}
