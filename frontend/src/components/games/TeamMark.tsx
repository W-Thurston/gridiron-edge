type TeamMarkProps = {
  abbr: string;
  /** Optional team color for tinting. Not used in T2. */
  color?: string;
};

export function TeamMark({ abbr, color }: TeamMarkProps) {
  return (
    <span
      className="team-mark"
      style={{
        background: color ?? "var(--bg-3)",
        color: color ? "var(--bg)" : "var(--ink)",
      }}
    >
      {abbr}
    </span>
  );
}
