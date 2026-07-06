type TeamHeroTeam = {
  /** Short abbreviation, e.g. "KAN" */
  abbr: string;
  /** City, e.g. "Kansas City" */
  city?: string | null;
  /** Team name, e.g. "Chiefs" */
  name?: string | null;
  /** Primary color hex, e.g. "#E31837". Falls back to grey. */
  primary_color?: string | null;
  /** Conference, e.g. "AFC" */
  conference?: string | null;
  /** Division letter, e.g. "W" */
  division?: string | null;
};

type TeamHeroProps = {
  /** Team metadata to render. */
  team: TeamHeroTeam;
  /** Optional context label above the team (e.g., "AWAY", "HOME", "AFC #1 Power"). */
  context?: string | null;
  /** Season record, e.g., "9-2". */
  record?: string | null;
  /** Team rating, e.g., 92.4. */
  rating?: number | null;
  /** ATS record, e.g., "7-4" (currently blocked; leave undefined). */
  atsRecord?: string | null;
  /** Layout direction. Left = mark then text; right = text then mark. */
  orientation?: "left" | "right";
  /** Team mark size in px. Default 56. */
  size?: number;
};

/**
 * Composed team identity block. Used as the large header on GameDetail
 * (two heroes side-by-side, one left, one right) and as the header band
 * on TeamProfile (one hero, always left).
 *
 * Renders:
 * - Team mark (colored square with abbreviation, `size` px)
 * - Optional context label above (small mono uppercase)
 * - City + name in serif (name italic)
 * - Optional record + ATS + rating meta row
 *
 * Layout direction via `orientation`. Default "left" for the common
 * single-team header case.
 */
export function TeamHero({
  team,
  context,
  record,
  rating,
  atsRecord,
  orientation = "left",
  size = 56,
}: TeamHeroProps) {
  const primaryColor = team.primary_color ?? "var(--bg-3)";
  const textColor = team.primary_color ? "#fff" : "var(--ink)";

  const mark = (
    <div
      style={{
        width: size,
        height: size,
        background: primaryColor,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--f-mono)",
        fontSize: size * 0.32,
        fontWeight: 700,
        color: textColor,
        borderRadius: 6,
        flexShrink: 0,
      }}
    >
      {team.abbr}
    </div>
  );

  const info = (
    <div style={{ textAlign: orientation === "right" ? "right" : "left" }}>
      {context && (
        <div
          className="mono upper dim"
          style={{ fontSize: 9.5, letterSpacing: "0.1em", marginBottom: 2 }}
        >
          {context}
          {team.conference && team.division && (
            <>
              {" · "}
              {team.conference} {team.division}
            </>
          )}
        </div>
      )}
      <div
        style={{
          fontFamily: "var(--f-serif)",
          fontSize: 24,
          fontWeight: 400,
          color: "var(--ink)",
          lineHeight: 1.1,
        }}
      >
        {team.city}{" "}
        <span style={{ fontStyle: "italic" }}>{team.name}</span>
      </div>
      {(record || rating != null || atsRecord) && (
        <div
          className="mono"
          style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 4 }}
        >
          {record && <>{record}</>}
          {atsRecord && (
            <>
              {record && " · "}
              ATS {atsRecord}
            </>
          )}
          {rating != null && (
            <>
              {(record || atsRecord) && " · "}
              Rating{" "}
              <span style={{ color: "var(--ink)" }}>
                {typeof rating === "number" ? rating.toFixed(1) : rating}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        justifyContent: orientation === "right" ? "flex-end" : "flex-start",
      }}
    >
      {orientation === "right" ? (
        <>
          {info}
          {mark}
        </>
      ) : (
        <>
          {mark}
          {info}
        </>
      )}
    </div>
  );
}
