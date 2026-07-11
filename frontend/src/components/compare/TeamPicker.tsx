import { useTeamRankings } from "../../api/hooks";
import { useTeamByAbbr } from "../../api/team_metadata_hook";
import { PendingField } from "../field-status/PendingField";
import { TeamMark } from "../primitives/TeamMark";

type TeamPickerProps = {
  teamA: string;
  teamB: string;
  onTeamAChange: (value: string) => void;
  onTeamBChange: (value: string) => void;
};

const TEAMS = [
  { value: "", label: "Select team…" },
  { value: "ARI", label: "Arizona Cardinals" },
  { value: "ATL", label: "Atlanta Falcons" },
  { value: "BAL", label: "Baltimore Ravens" },
  { value: "BUF", label: "Buffalo Bills" },
  { value: "CAR", label: "Carolina Panthers" },
  { value: "CHI", label: "Chicago Bears" },
  { value: "CIN", label: "Cincinnati Bengals" },
  { value: "CLE", label: "Cleveland Browns" },
  { value: "DAL", label: "Dallas Cowboys" },
  { value: "DEN", label: "Denver Broncos" },
  { value: "DET", label: "Detroit Lions" },
  { value: "GB", label: "Green Bay Packers" },
  { value: "HOU", label: "Houston Texans" },
  { value: "IND", label: "Indianapolis Colts" },
  { value: "JAC", label: "Jacksonville Jaguars" },
  { value: "KAN", label: "Kansas City Chiefs" },
  { value: "LAC", label: "Los Angeles Chargers" },
  { value: "LAR", label: "Los Angeles Rams" },
  { value: "LV", label: "Las Vegas Raiders" },
  { value: "MIA", label: "Miami Dolphins" },
  { value: "MIN", label: "Minnesota Vikings" },
  { value: "NE", label: "New England Patriots" },
  { value: "NO", label: "New Orleans Saints" },
  { value: "NYG", label: "New York Giants" },
  { value: "NYJ", label: "New York Jets" },
  { value: "PHI", label: "Philadelphia Eagles" },
  { value: "PIT", label: "Pittsburgh Steelers" },
  { value: "SEA", label: "Seattle Seahawks" },
  { value: "SF", label: "San Francisco 49ers" },
  { value: "TB", label: "Tampa Bay Buccaneers" },
  { value: "TEN", label: "Tennessee Titans" },
  { value: "WAS", label: "Washington Commanders" },
] as const;

/**
 * Enhanced team picker. Two picker cards (team-colored mark, dropdown,
 * rating, record, pending-marked Off/Def stats) framing a center swap
 * button. External API preserved: teamA/teamB + onChange callbacks.
 */
export function TeamPicker({
  teamA,
  teamB,
  onTeamAChange,
  onTeamBChange,
}: TeamPickerProps) {
  const rankings = useTeamRankings();
  const items = rankings.data?.items ?? [];

  const findTeam = (abbr: string) =>
    items.find((t) => t.abbr === abbr) ?? null;

  const swap = () => {
    const prevA = teamA;
    onTeamAChange(teamB);
    onTeamBChange(prevA);
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(240px, 340px) auto minmax(240px, 340px)",
        gap: 16,
        alignItems: "stretch",
        justifyContent: "center",
      }}
    >
      <PickerCard
        label="Team A"
        selected={teamA}
        onChange={onTeamAChange}
        ranking={findTeam(teamA)}
        orientation="right"
      />

      {/* Center: vs + swap */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
        }}
      >
        <span
          className="serif"
          style={{ fontSize: 20, fontStyle: "italic", color: "var(--ink-2)" }}
        >
          vs
        </span>
        <button
          type="button"
          onClick={swap}
          aria-label="Swap teams"
          title="Swap teams"
          disabled={!teamA && !teamB}
          style={{
            background: "transparent",
            border: "1px solid var(--line-soft)",
            borderRadius: 4,
            padding: "4px 8px",
            cursor: teamA || teamB ? "pointer" : "not-allowed",
            color: "var(--ink-3)",
            fontSize: 14,
            fontFamily: "var(--f-mono)",
            lineHeight: 1,
          }}
        >
          ⇄
        </button>
      </div>

      <PickerCard
        label="Team B"
        selected={teamB}
        onChange={onTeamBChange}
        ranking={findTeam(teamB)}
        orientation="left"
      />
    </div>
  );
}

/**
 * Single team picker card: dropdown + (when selected) team-colored mark,
 * name, rating, record, and pending-marked Off/Def stats.
 *
 * Orientation mirrors the card's content toward the center:
 * - "right" (Team A): label + text right-aligned, identity reads
 *   name → logo (logo on the inside/right edge)
 * - "left" (Team B): label + text left-aligned, identity reads
 *   logo → name (logo on the inside/left edge)
 */
function PickerCard({
  label,
  selected,
  onChange,
  ranking,
  orientation,
}: {
  label: string;
  selected: string;
  onChange: (value: string) => void;
  ranking: {
    abbr: string;
    name: string;
    rating?: number | null;
    rank?: number | null;
    record?: { wins: number; losses: number; ties: number } | null;
  } | null;
  orientation: "left" | "right";
}) {
  const metadata = useTeamByAbbr(selected);
  const primaryColor = metadata?.primary_color;

  const background = primaryColor
    ? `linear-gradient(180deg, color-mix(in oklab, ${primaryColor} 22%, var(--bg-1)) 0%, var(--bg-1) 100%)`
    : "var(--bg-1)";

  const recordText = ranking?.record
    ? `${ranking.record.wins}-${ranking.record.losses}${
        ranking.record.ties > 0 ? `-${ranking.record.ties}` : ""
      }`
    : null;

  const alignRight = orientation === "right";
  const textAlign = alignRight ? "right" : "left";

  return (
    <div
      className="hm-card"
      style={{
        padding: 16,
        background: selected ? background : "var(--bg-1)",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        textAlign,
      }}
    >
      <span className="upper dim2" style={{ fontSize: 9 }}>
        {label}
      </span>

      {/* Dropdown */}
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        style={selectStyle}
      >
        {TEAMS.map((team) => (
          <option key={team.value} value={team.value}>
            {team.label}
          </option>
        ))}
      </select>

      {/* Team identity when selected — mirrored by orientation */}
      {selected && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            flexDirection: alignRight ? "row-reverse" : "row",
            justifyContent: alignRight ? "flex-start" : "flex-start",
          }}
        >
          <TeamMark abbr={selected} size={36} />
          <div style={{ textAlign }}>
            <div style={{ fontWeight: 500, fontSize: 13 }}>
              {ranking?.name ?? selected}
            </div>
            <div
              className="mono"
              style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 2 }}
            >
              {recordText && <>{recordText}</>}
              {ranking?.rating != null && (
                <>
                  {recordText && " · "}
                  Rating{" "}
                  <span style={{ color: "var(--ink)" }}>
                    {ranking.rating.toFixed(0)}
                  </span>
                </>
              )}
              {ranking?.rank != null && (
                <>
                  {" · "}#{ranking.rank}
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Off/Def mini-stats (pending — off/def decomposition blocked) */}
      {selected && (
        <div
          style={{
            display: "flex",
            gap: 16,
            fontSize: 10,
            color: "var(--ink-4)",
            justifyContent: alignRight ? "flex-end" : "flex-start",
          }}
          className="mono"
        >
          <span
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            Off <PendingField placeholder="" />
          </span>
          <span
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            Def <PendingField placeholder="" />
          </span>
        </div>
      )}
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: "var(--bg-1)",
  color: "var(--ink)",
  border: "1px solid var(--line-soft)",
  borderRadius: 5,
  padding: "4px 8px",
  fontSize: 12,
  fontFamily: "var(--f-sans)",
  width: "100%",
};
