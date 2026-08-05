import { useTeamByAbbr } from "../../api/team_metadata_hook";

type TeamMarkProps = {
  abbr: string;
  size?: number;
};

/**
 * Team abbreviation chip with team primary color background.
 *
 * Consumes team metadata from React Query cache. Falls back to
 * a neutral grey background when color is unavailable (cache empty,
 * team not found, or backend returned null).
 *
 * Sized via optional prop; default 22px matches the prototype
 * convention.
 */
export function TeamMark({ abbr, size = 22 }: TeamMarkProps) {
  const team = useTeamByAbbr(abbr);
  const displayAbbr = team?.abbr ?? abbr;
  const primaryColor = team?.primary_color;

  const bg = primaryColor ?? "var(--bg-3)";
  const textColor = primaryColor ? "#fff" : "var(--ink)";

  return (
    <span
      className="team-mark mono"
      style={{
        background: bg,
        color: textColor,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        fontSize: Math.max(9, size * 0.45),
        fontWeight: 600,
        borderRadius: 3,
        letterSpacing: "0.02em",
      }}
    >
      {displayAbbr}
    </span>
  );
}
