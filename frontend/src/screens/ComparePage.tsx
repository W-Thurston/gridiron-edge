import { useEffect, useState } from "react";
import { useCompareTeams, useTeamProfile, usePlayersList, usePlayerHistory, useDefenseAllowed } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/primitives/TeamMark";
import { Pill } from "../components/primitives/Pill";
import { TeamPicker } from "../components/compare/TeamPicker";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";
import { usePendingHighlight } from "../components/field-status/usePendingHighlight";
import { BarChart } from "../components/primitives/BarChart";
import { PendingChip } from "../components/field-status/PendingChip";

type CompareMode = "team" | "player";

export function ComparePage() {
  const { route, navigate } = useNav();
  const mode: CompareMode = route.params.mode === "player" ? "player" : "team";

  const setMode = (newMode: CompareMode) => {
    const params: Record<string, string> = { mode: newMode };
    if (route.params.team_a) params.team_a = route.params.team_a;
    if (route.params.team_b) params.team_b = route.params.team_b;
    navigate("/compare", params);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
        maxWidth: 920,
        margin: "0 auto",
      }}
    >
      {/* Mode switcher */}
      <div style={{ display: "flex", gap: 6 }}>
        <Pill active={mode === "team"} onClick={() => setMode("team")}>
          Team vs Team
        </Pill>
        <Pill active={mode === "player"} onClick={() => setMode("player")}>
          Player vs Defense
        </Pill>
      </div>

      {/* Mode content */}
      {mode === "team" ? <TeamCompareMode /> : <PlayerCompareMode />}
    </div>
  );
}

/**
 * Team vs Team comparison mode. Preserves existing pickers + stat table.
 * Tier 2 rebuilds internals into grouped matchup sections + narrative.
 */
function TeamCompareMode() {
  const { route, navigate } = useNav();
  const initialTeamA = route.params.team_a ?? "";
  const initialTeamB = route.params.team_b ?? "";

  const [teamA, setTeamA] = useState(initialTeamA);
  const [teamB, setTeamB] = useState(initialTeamB);
  const [cohort, setCohort] = useState<CohortKey>("season");

  useEffect(() => {
    const params: Record<string, string> = { mode: "team" };
    if (teamA) params.team_a = teamA;
    if (teamB) params.team_b = teamB;
    navigate("/compare", params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teamA, teamB]);

  const { data, isLoading, error, refetch } = useCompareTeams({
    team_a: teamA || null,
    team_b: teamB || null,
  });

  const profileA = useTeamProfile(teamA || null);
  const profileB = useTeamProfile(teamB || null);

  const bothSelected = teamA && teamB;
  const cohortA = extractCohort(profileA.data?.cohort_splits, cohort);
  const cohortB = extractCohort(profileB.data?.cohort_splits, cohort);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Team pickers (float directly — no outer card wrapper) */}
      <TeamPicker
        teamA={teamA}
        teamB={teamB}
        onTeamAChange={setTeamA}
        onTeamBChange={setTeamB}
      />

      {/* Cohort strip (floating, below pickers) */}
      {bothSelected && data && (
        <div
          style={{
            display: "flex",
            gap: 6,
            alignItems: "center",
            padding: "0 4px",
          }}
        >
          <span className="upper dim2" style={{ fontSize: 9, marginRight: 6 }}>
            Cohort
          </span>
          {COHORT_TABS.map((tab) => (
            <Pill
              key={tab.key}
              active={cohort === tab.key}
              onClick={() => setCohort(tab.key)}
            >
              {tab.label}
            </Pill>
          ))}
          <span
            className="mono dim2"
            style={{ fontSize: 10, marginLeft: "auto" }}
          >
            Season: {data.season ?? "—"}
          </span>
        </div>
      )}

      {/* Empty / loading / error states */}
      {!bothSelected && (
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim mono" style={{ fontSize: 12 }}>
            Select two teams to compare.
          </div>
        </div>
      )}

      {bothSelected && isLoading && (
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">Loading…</div>
        </div>
      )}

      {bothSelected && error && (
        <ErrorCard
          error={error}
          onRetry={() => refetch()}
          title="Couldn't load comparison"
        />
      )}

      {/* Content cards */}
      {bothSelected && data && (
        <>
          {/* Narrative card */}
          {cohortA && cohortB && (
            <div className="hm-card" style={{ padding: 20 }}>
              <NarrativeBanner
                teamA={data.team_a}
                teamB={data.team_b}
                cohortA={cohortA}
                cohortB={cohortB}
              />
            </div>
          )}

          {/* Collapsible summary card */}
          <SummaryCard data={data} />

          {/* Matchup cards */}
          {cohortA && cohortB ? (
            <>
              <div className="hm-card" style={{ padding: 20 }}>
                <BallGroup
                  title={`When ${data.team_a} has the ball`}
                  subtitle={`${data.team_a} offense vs ${data.team_b} defense`}
                  offCohort={cohortA}
                  defCohort={cohortB}
                  offTeam={data.team_a}
                  defTeam={data.team_b}
                />
              </div>
              <div className="hm-card" style={{ padding: 20 }}>
                <BallGroup
                  title={`When ${data.team_b} has the ball`}
                  subtitle={`${data.team_b} offense vs ${data.team_a} defense`}
                  offCohort={cohortB}
                  defCohort={cohortA}
                  offTeam={data.team_b}
                  defTeam={data.team_a}
                />
              </div>
              <div className="hm-card" style={{ padding: 20 }}>
                <EvenFooting cohortA={cohortA} cohortB={cohortB} />
              </div>
            </>
          ) : (
            <div className="hm-card" style={{ padding: 24 }}>
              <div
                style={{
                  textAlign: "center",
                  color: "var(--ink-4)",
                  fontSize: 12,
                }}
              >
                No cohort split data for this selection.
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Grid template for summary rows: A value | stat label | B value. */
const SUMMARY_GRID = "150px 200px 150px";

/**
 * Collapsible summary card — non-cohort stats (rating/rank/record/
 * percentiles). Centered 3-column layout matching the matchup cards'
 * team-left / team-right format, but without fill bars or collision
 * coloring (these are parallel values, not collisions).
 */
function SummaryCard({
  data,
}: {
  data: {
    team_a: string;
    team_b: string;
    stats?: Array<{
      key: string;
      label: string;
      team_a_value?: number | string | null;
      team_b_value?: number | string | null;
    }> | null;
    _meta?: { field_status?: Record<string, unknown> } | null;
  };
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          width: "100%",
          background: "transparent",
          border: "none",
          padding: 0,
          cursor: "pointer",
          font: "inherit",
        }}
      >
        <span className="upper dim" style={{ fontSize: 10 }}>
          Summary Stats
        </span>
        <span className="mono dim2" style={{ fontSize: 12 }}>
          {open ? "▾" : "▸"}
        </span>
      </button>

      {open && (
        <div style={{ marginTop: 12 }}>
          {/* Team headers */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: SUMMARY_GRID,
              gap: 10,
              justifyContent: "center",
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            {/* Team A (right-aligned, name → mark) */}
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                justifyContent: "flex-end",
                fontWeight: 500,
                color: "var(--ink)",
              }}
            >
              {data.team_a}
              <TeamMark abbr={data.team_a} size={18} />
            </span>
            <span />
            {/* Team B (left-aligned, mark → name) */}
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                justifyContent: "flex-start",
                fontWeight: 500,
                color: "var(--ink)",
              }}
            >
              <TeamMark abbr={data.team_b} size={18} />
              {data.team_b}
            </span>
          </div>

          {/* Stat rows */}
          <div style={{ display: "grid", gap: 2 }}>
            {(data.stats ?? []).map((row, i) => (
              <SummaryRow
                key={row.key}
                row={row}
                first={i === 0}
                status={
                  data._meta?.field_status?.[row.key] as
                    | FieldStatus
                    | undefined
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * One summary stat row: A value (right) | centered stat label | B value
 * (left). No bars/coloring — parallel values, not a collision.
 */
function SummaryRow({
  row,
  first,
  status,
}: {
  row: {
    key: string;
    label: string;
    team_a_value?: number | string | null;
    team_b_value?: number | string | null;
  };
  first: boolean;
  status: FieldStatus | undefined;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: SUMMARY_GRID,
        gap: 10,
        justifyContent: "center",
        alignItems: "center",
        padding: "9px 0",
        borderTop: first ? "none" : "1px solid var(--line-soft)",
        fontSize: 12,
      }}
    >
      {/* Team A value (right-aligned toward center) */}
      <span
        className="mono tnum"
        style={{ textAlign: "right", color: "var(--ink)" }}
      >
        <CompareCell value={row.team_a_value} status={status} />
      </span>

      {/* Stat label (center) */}
      <span
        className="dim"
        style={{
          textAlign: "center",
          fontSize: 10.5,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
        }}
      >
        {row.label}
        <StatRowLabelStatus status={status} />
      </span>

      {/* Team B value (left-aligned toward center) */}
      <span
        className="mono tnum"
        style={{ textAlign: "left", color: "var(--ink-2)" }}
      >
        <CompareCell value={row.team_b_value} status={status} />
      </span>
    </div>
  );
}

type CohortKey = "season" | "l4" | "home" | "away";

const COHORT_TABS: { key: CohortKey; label: string }[] = [
  { key: "season", label: "Season" },
  { key: "l4", label: "Last 4" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
];

/** Extract a single cohort's metric dict from a team's cohort_splits. */
function extractCohort(
  cohortSplits: { [key: string]: unknown } | null | undefined,
  cohort: CohortKey,
): Record<string, number> | null {
  if (!cohortSplits) return null;
  const c = cohortSplits[cohort] as Record<string, number> | undefined;
  return c ?? null;
}

type MatchupMetric = {
  off: string;
  def: string;
  title: string; // center title, e.g. "Run efficiency"
  offLabel: string; // e.g. "Rush EPA / play"
  defLabel: string; // e.g. "Rush EPA allowed / play"
  fmt: (v: number) => string;
};

const epaFmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(3);
const pctFmt = (v: number) => (v * 100).toFixed(1) + "%";

const MATCHUP_METRICS: MatchupMetric[] = [
  {
    off: "off_epa_per_play", def: "def_epa_per_play",
    title: "Overall efficiency",
    offLabel: "EPA / play", defLabel: "EPA allowed / play",
    fmt: epaFmt,
  },
  {
    off: "off_pass_epa", def: "def_pass_epa",
    title: "Pass efficiency",
    offLabel: "Pass EPA / play", defLabel: "Pass EPA allowed / play",
    fmt: epaFmt,
  },
  {
    off: "off_rush_epa", def: "def_rush_epa",
    title: "Run efficiency",
    offLabel: "Rush EPA / play", defLabel: "Rush EPA allowed / play",
    fmt: epaFmt,
  },
  {
    off: "off_third_down_pct", def: "def_third_down_pct",
    title: "Third down",
    offLabel: "3rd-down conv %", defLabel: "3rd-down allowed %",
    fmt: pctFmt,
  },
  {
    off: "off_redzone_td_pct", def: "def_redzone_td_pct",
    title: "Red zone",
    offLabel: "Red-zone TD %", defLabel: "Red-zone TD allowed %",
    fmt: pctFmt,
  },
];

/** Ordinal suffix: 1 → "1st", 3 → "3rd", 16 → "16th". */
function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}

/**
 * Horizontal rank-fill bar. Fill fraction = (33 - rank) / 32, so rank 1
 * is full and rank 32 is nearly empty. Anchored left or right so pairs
 * of bars can mirror toward the center.
 */
function RankBar({
  rank,
  anchor,
  color,
}: {
  rank: number | null | undefined;
  anchor: "left" | "right";
  color: string;
}) {
  const fill =
    rank != null ? Math.max(0, Math.min(1, (33 - rank) / 32)) : 0;

  const fillStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: `${fill * 100}%`,
    background: color,
    borderRadius: 3,
  };
  if (anchor === "left") fillStyle.left = 0;
  else fillStyle.right = 0;

  return (
    <div
      style={{
        height: 6,
        background: "var(--bg-3)",
        position: "relative",
        borderRadius: 3,
      }}
    >
      <div style={fillStyle} />
    </div>
  );
}

/**
 * Edge chip for the center column. Arrow points toward the favored
 * side; shows team + descriptor. Green when there's a real edge, dim
 * when even.
 */
function EdgeChip({
  edge,
  offTeam,
  defTeam,
}: {
  edge: number;
  offTeam: string;
  defTeam: string;
}) {
  const descriptor = edgeDescriptor(edge);
  if (descriptor === "Even") {
    return (
      <span
        className="mono upper"
        style={{
          fontSize: 8.5,
          color: "var(--ink-4)",
          letterSpacing: "0.06em",
        }}
      >
        — even —
      </span>
    );
  }
  const offenseFavored = edge > 0;
  const favored = offenseFavored ? offTeam : defTeam;
  const arrow = offenseFavored ? "◄" : "►";
  return (
    <span
      className="mono upper"
      style={{
        fontSize: 8.5,
        color: "var(--pos)",
        background: "color-mix(in oklab, var(--pos) 12%, transparent)",
        padding: "2px 6px",
        borderRadius: 3,
        letterSpacing: "0.05em",
        whiteSpace: "nowrap",
      }}
    >
      {offenseFavored ? `${arrow} ${favored}` : `${favored} ${arrow}`}{" "}
      {descriptor}
    </span>
  );
}

/** Shared 5-column grid template for matchup rows. */
const MATCHUP_GRID = "120px 90px 150px 90px 120px";

/** One directional group: offense metrics vs reciprocal defense-allowed. */
function BallGroup({
  title,
  subtitle,
  offCohort,
  defCohort,
  offTeam,
  defTeam,
}: {
  title: string;
  subtitle: string;
  offCohort: Record<string, number>;
  defCohort: Record<string, number>;
  offTeam: string;
  defTeam: string;
}) {
  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600 }}>{title}</div>
        <div className="dim mono" style={{ fontSize: 10.5 }}>
          {subtitle}
        </div>
      </div>

      {/* Column headers */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: MATCHUP_GRID,
          gap: 10,
          justifyContent: "center",
          fontSize: 9,
          color: "var(--ink-4)",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 6,
          alignItems: "center",
        }}
      >
        <span style={{ textAlign: "right" }}>{offTeam} offense</span>
        <span />
        <span />
        <span />
        <span style={{ textAlign: "left" }}>{defTeam} defense</span>
      </div>

      <div style={{ display: "grid", gap: 2 }}>
        {MATCHUP_METRICS.map((m, i) => {
          const offVal = offCohort[m.off];
          const defVal = defCohort[m.def];
          const offRank = offCohort[`rank_${m.off}`];
          const defRank = defCohort[`rank_${m.def}`];
          const edge =
            offRank != null && defRank != null ? defRank - offRank : 0;
          const offColor = edge > 0 ? "var(--pos)" : "var(--ink-4)";
          const defColor = edge < 0 ? "var(--pos)" : "var(--ink-4)";

          return (
            <div
                key={m.off}
                style={{
                  display: "grid",
                  gridTemplateColumns: MATCHUP_GRID,
                  gap: 10,
                  justifyContent: "center",
                  alignItems: "center",
                  padding: "10px 0",
                  borderTop: i === 0 ? "none" : "1px solid var(--line-soft)",
                }}
              >
                {/* Offense value + rank (same line) + sublabel */}
                <div style={{ textAlign: "right" }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "baseline",
                      justifyContent: "flex-end",
                      gap: 5,
                    }}
                  >
                    <span
                      className="mono tnum"
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: edge > 0 ? "var(--pos)" : "var(--ink)",
                      }}
                    >
                      {offVal != null ? m.fmt(offVal) : "—"}
                    </span>
                    {offRank != null && (
                      <span className="mono dim2" style={{ fontSize: 9.5 }}>
                        {ordinal(offRank)}
                      </span>
                    )}
                  </div>
                  <div className="dim2" style={{ fontSize: 9, marginTop: 1 }}>
                    {m.offLabel}
                  </div>
                </div>

                {/* Offense bar */}
                <RankBar rank={offRank} anchor="right" color={offColor} />

                {/* Center: title + edge chip */}
                <div style={{ textAlign: "center" }}>
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 500,
                      color: "var(--ink-2)",
                      marginBottom: 3,
                    }}
                  >
                    {m.title}
                  </div>
                  <EdgeChip edge={edge} offTeam={offTeam} defTeam={defTeam} />
                </div>

                {/* Defense bar */}
                <RankBar rank={defRank} anchor="left" color={defColor} />

                {/* Defense value + rank (same line) + sublabel */}
                <div style={{ textAlign: "left" }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "baseline",
                      justifyContent: "flex-start",
                      gap: 5,
                    }}
                  >
                    <span
                      className="mono tnum"
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: edge < 0 ? "var(--pos)" : "var(--ink)",
                      }}
                    >
                      {defVal != null ? m.fmt(defVal) : "—"}
                    </span>
                    {defRank != null && (
                      <span className="mono dim2" style={{ fontSize: 9.5 }}>
                        {ordinal(defRank)}
                      </span>
                    )}
                  </div>
                  <div className="dim2" style={{ fontSize: 9, marginTop: 1 }}>
                    {m.defLabel}
                  </div>
                </div>
              </div>
          );
        })}
      </div>
    </div>
  );
}

/** Neutral section: turnover_diff head-to-head, same bar language. */
function EvenFooting({
  cohortA,
  cohortB,
}: {
  cohortA: Record<string, number>;
  cohortB: Record<string, number>;
}) {
  const aVal = cohortA["turnover_diff"];
  const bVal = cohortB["turnover_diff"];
  const aRank = cohortA["rank_turnover_diff"];
  const bRank = cohortB["rank_turnover_diff"];
  const fmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(3);

  // Lower rank wins (rank 1 = best). edge = bRank - aRank; positive = A better.
  const edge = aRank != null && bRank != null ? bRank - aRank : 0;
  const aColor = edge > 0 ? "var(--pos)" : "var(--ink-4)";
  const bColor = edge < 0 ? "var(--pos)" : "var(--ink-4)";

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600 }}>Even footing</div>
        <div className="dim mono" style={{ fontSize: 10.5 }}>
          Neutral metrics
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: MATCHUP_GRID,
          gap: 10,
          justifyContent: "center",
          alignItems: "center",
          padding: "10px 0",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "flex-end",
            gap: 5,
          }}
        >
          <span
            className="mono tnum"
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: edge > 0 ? "var(--pos)" : "var(--ink)",
            }}
          >
            {aVal != null ? fmt(aVal) : "—"}
          </span>
          {aRank != null && (
            <span className="mono dim2" style={{ fontSize: 9.5 }}>
              {ordinal(aRank)}
            </span>
          )}
        </div>

        <RankBar rank={aRank} anchor="right" color={aColor} />

        <div className="dim" style={{ fontSize: 10, textAlign: "center" }}>
          Turnover diff
        </div>

        <RankBar rank={bRank} anchor="left" color={bColor} />

        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "flex-start",
            gap: 5,
          }}
        >
          <span
            className="mono tnum"
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: edge < 0 ? "var(--pos)" : "var(--ink)",
            }}
          >
            {bVal != null ? fmt(bVal) : "—"}
          </span>
          {bRank != null && (
            <span className="mono dim2" style={{ fontSize: 9.5 }}>
              {ordinal(bRank)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

type PlayerStatOption = { label: string; statKey: string; statType: string };

const POSITION_STATS: Record<string, PlayerStatOption[]> = {
  QB: [
    { label: "Passing", statKey: "pass_yards", statType: "qb_pass_yards" },
    { label: "Rushing", statKey: "rush_yards", statType: "qb_rush_yards" },
  ],
  RB: [{ label: "Rushing", statKey: "rush_yards", statType: "rb_rush_yards" }],
  WR: [{ label: "Receiving", statKey: "rec_yards", statType: "wr_rec_yards" }],
  TE: [{ label: "Receiving", statKey: "rec_yards", statType: "te_rec_yards" }],
  FB: [{ label: "Rushing", statKey: "rush_yards", statType: "rb_rush_yards" }],
};

type PlayerSplitKey =
  | "season" | "l4" | "home" | "away"
  | "vs_winning" | "vs_losing" | "vs_top10";

const PLAYER_SPLITS: { key: PlayerSplitKey; label: string; pending: boolean }[] = [
  { key: "season", label: "Season", pending: false },
  { key: "l4", label: "Last 4", pending: false },
  { key: "home", label: "Home", pending: false },
  { key: "away", label: "Away", pending: false },
  { key: "vs_winning", label: "vs Winning", pending: true },
  { key: "vs_losing", label: "vs Losing", pending: true },
  { key: "vs_top10", label: "vs Top-10", pending: true },
];

/**
 * Player vs Defense mode. Independent player / stat-category / team
 * pickers (mirroring Team-vs-Team), a 7-split strip (4 live + 3 pending),
 * and placeholder sections for the bar chart (C2) and matchup card +
 * comparison table (C3).
 */
function PlayerCompareMode() {
  const { route, navigate } = useNav();
  const playersResult = usePlayersList({});
  const players = playersResult.data?.items ?? [];

  const playerId = route.params.player_id ?? "";
  const statKey = route.params.stat ?? "";
  const team = route.params.team ?? "";
  const split = (route.params.split ?? "season") as PlayerSplitKey;

  const selectedPlayer = players.find((p) => p.player_id === playerId) ?? null;
  const statOptions = selectedPlayer
    ? (POSITION_STATS[selectedPlayer.position] ?? [])
    : [];
  const selectedStat = statOptions.find((s) => s.statKey === statKey) ?? null;

  const setParams = (patch: Record<string, string>) => {
    const next: Record<string, string> = {
      mode: "player",
      ...(playerId && { player_id: playerId }),
      ...(statKey && { stat: statKey }),
      ...(team && { team }),
      split,
      ...patch,
    };
    // Drop empties so the URL stays clean.
    for (const k of Object.keys(next)) if (!next[k]) delete next[k];
    navigate("/compare", next);
  };

  const onSelectPlayer = (id: string) => {
    // Reset stat when player changes (position may differ).
    const p = players.find((x) => x.player_id === id);
    const firstStat = p ? (POSITION_STATS[p.position]?.[0]?.statKey ?? "") : "";
    setParams({ player_id: id, stat: firstStat });
  };

  const ready = selectedPlayer && selectedStat && team;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Pickers row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(260px, 360px) auto minmax(200px, 300px)",
          gap: 16,
          alignItems: "stretch",
          justifyContent: "center",
        }}
      >
        {/* Left card: player + stat */}
        <div className="hm-card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="upper dim2" style={{ fontSize: 9 }}>Player</span>
          <PlayerCombobox
            players={players}
            selected={selectedPlayer}
            onSelect={onSelectPlayer}
          />
          <span className="upper dim2" style={{ fontSize: 9, marginTop: 4 }}>Stat</span>
          <select
            value={statKey}
            disabled={!selectedPlayer}
            onChange={(e) => setParams({ stat: e.target.value })}
            style={selectStyle}
          >
            {statOptions.length === 0 && <option value="">—</option>}
            {statOptions.map((s) => (
              <option key={s.statKey} value={s.statKey}>{s.label}</option>
            ))}
          </select>
        </div>

        {/* Center vs */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span className="serif" style={{ fontSize: 20, fontStyle: "italic", color: "var(--ink-2)" }}>
            vs
          </span>
        </div>

        {/* Right card: team */}
        <div className="hm-card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="upper dim2" style={{ fontSize: 9 }}>Defense</span>
          <select
            value={team}
            onChange={(e) => setParams({ team: e.target.value })}
            style={selectStyle}
          >
            {PLAYER_TEAMS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
          {team && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
              <TeamMark abbr={team} size={28} />
              <span className="mono dim" style={{ fontSize: 11 }}>{team} defense</span>
            </div>
          )}
        </div>
      </div>

      {/* Split strip */}
      {ready && (
        <div style={{ display: "flex", gap: 6, alignItems: "center", padding: "0 4px" }}>
          <span className="upper dim2" style={{ fontSize: 9, marginRight: 6 }}>Split</span>
          {PLAYER_SPLITS.map((s) =>
            s.pending ? (
              <PendingSplitPill key={s.key} label={s.label} />
            ) : (
              <Pill
                key={s.key}
                active={split === s.key}
                onClick={() => setParams({ split: s.key })}
              >
                {s.label}
              </Pill>
            ),
          )}
        </div>
      )}

      {/* Empty state */}
      {!ready && (
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim mono" style={{ fontSize: 12 }}>
            Select a player, stat, and defense to compare.
          </div>
        </div>
      )}

      {/* Placeholders — C2 bar chart, C3 matchup card + table */}
      {ready && (
        <>
          <PlayerBarChartCard
            playerId={selectedPlayer.player_id}
            playerName={selectedPlayer.player_name}
            statLabel={selectedStat.label}
            statKey={selectedStat.statKey}
            statType={selectedStat.statType}
            team={team}
            split={split}
          />
          <MatchupPlainlyCard
            playerId={selectedPlayer.player_id}
            playerName={selectedPlayer.player_name}
            statLabel={selectedStat.label}
            statKey={selectedStat.statKey}
            statType={selectedStat.statType}
            team={team}
            split={split}
          />
        </>
      )}
    </div>
  );
}

type CohortAllowed = {
  avg_allowed?: number | null;
  sample_size?: number | null;
  rank_against_position?: number | null;
};

/** Mean of numeric values, or null if none. */
function mean(vals: (number | null | undefined)[]): number | null {
  const nums = vals.filter((v): v is number => typeof v === "number");
  if (nums.length === 0) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

/** Player's stat average for a given live split, from B1 game rows. */
function playerAvgForSplit(
  games: { week: number; value: number | null; is_home: boolean }[],
  split: PlayerSplitKey,
): number | null {
  if (split === "season") return mean(games.map((g) => g.value));
  if (split === "l4") {
    const last4 = [...games].sort((a, b) => a.week - b.week).slice(-4);
    return mean(last4.map((g) => g.value));
  }
  if (split === "home") return mean(games.filter((g) => g.is_home).map((g) => g.value));
  if (split === "away") return mean(games.filter((g) => !g.is_home).map((g) => g.value));
  return null; // pending splits
}

/**
 * C3 — "The matchup, plainly" verdict card + comparison table.
 *
 * Verdict from defense rank tier + a player-baseline comparison. Table
 * shows player avg (from B1 bars, per split) vs defense-allowed + rank
 * (from B3) across the 4 live splits; 3 pending splits marked.
 */
function MatchupPlainlyCard({
  playerId,
  playerName,
  statLabel,
  statKey,
  statType,
  team,
  split,
}: {
  playerId: string;
  playerName: string;
  statLabel: string;
  statKey: string;
  statType: string;
  team: string;
  split: PlayerSplitKey;
}) {
  const history = usePlayerHistory(playerId, { stat: statKey });
  const defense = useDefenseAllowed(team, { stat_type: statType });

  const games = (history.data?.items ?? []).map((g) => ({
    week: g.week,
    value: g.value ?? null,
    is_home: Boolean(g.is_home),
  }));

  const cohorts = defense.data?.cohorts as
    | Record<string, CohortAllowed>
    | null
    | undefined;

  const current = cohorts?.[split];
  const allowed = current?.avg_allowed ?? null;
  const rank = current?.rank_against_position ?? null;
  const playerAvg = playerAvgForSplit(games, split);

  const loading = history.isLoading || defense.isLoading;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 16,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          The matchup, plainly
        </div>
        {history.data?.season != null && (
          <span className="mono dim2" style={{ fontSize: 9 }}>
            {history.data.season}-{history.data.season + 1}
          </span>
        )}
      </div>

      {loading ? (
        <div className="dim" style={{ padding: 12 }}>Loading…</div>
      ) : allowed == null ? (
        <div
          style={{
            padding: 16,
            textAlign: "center",
            color: "var(--ink-4)",
            fontSize: 12,
          }}
        >
          No defense-allowed data for this split.
        </div>
      ) : (
        <>
          {/* Big number + rank + verdict */}
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span
              className="mono tnum"
              style={{ fontSize: 30, fontWeight: 600, color: "var(--ink)" }}
            >
              {allowed.toFixed(1)}
            </span>
            <span className="dim" style={{ fontSize: 12 }}>
              {statLabel.toLowerCase()} allowed
            </span>
          </div>

          {rank != null && <RankLine rank={rank} />}

          <VerdictCallout playerAvg={playerAvg} allowed={allowed} />

          {/* Player baseline comparison */}
          {playerAvg != null && (
            <div
              className="mono"
              style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 10 }}
            >
              {playerName} averages{" "}
              <span style={{ color: "var(--ink)", fontWeight: 500 }}>
                {playerAvg.toFixed(1)}
              </span>{" "}
              ({split}); this defense allows{" "}
              <span style={{ color: "var(--ink)", fontWeight: 500 }}>
                {allowed.toFixed(1)}
              </span>{" "}
              <DeltaTag delta={allowed - playerAvg} />
            </div>
          )}

          {/* Comparison table across splits */}
          <ComparisonTable games={games} cohorts={cohorts ?? null} playerName={playerName} />
        </>
      )}
    </div>
  );
}

/** Rank line — general defensive context (not the player-specific verdict). */
function RankLine({ rank }: { rank: number }) {
  const descriptor =
    rank <= 10 ? "stingy" : rank >= 23 ? "generous" : "middle-of-pack";
  return (
    <div className="mono dim" style={{ fontSize: 11, marginTop: 4 }}>
      {ordinal(rank)} of 32 vs the position ({descriptor} overall)
    </div>
  );
}

/**
 * Verdict for THIS player: driven by the baseline delta (defense-allowed
 * vs the player's own average), not the defense's general rank. A defense
 * allowing MORE than the player's norm is a favorable spot (expect over);
 * allowing LESS is tough (expect under). Rank is shown separately as
 * general context.
 *
 * Deadband: within ±8% of the player's average → Neutral.
 */
function VerdictCallout({
  playerAvg,
  allowed,
}: {
  playerAvg: number | null;
  allowed: number;
}) {
  // Without a player baseline we can't judge favorable/tough for them.
  if (playerAvg == null || playerAvg === 0) {
    return (
      <div
        style={{
          marginTop: 12,
          padding: "10px 12px",
          borderRadius: 6,
          background: "var(--bg-2)",
          borderLeft: "3px solid var(--ink-3)",
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)" }}>
          Neutral matchup
        </div>
        <div className="dim" style={{ fontSize: 10.5, marginTop: 2 }}>
          No player baseline to compare against.
        </div>
      </div>
    );
  }

  const delta = allowed - playerAvg;
  const relative = delta / playerAvg;
  const deadband = 0.08;

  let verdict: string;
  let detail: string;
  let color: string;

  if (relative > deadband) {
    verdict = "Favorable spot";
    detail = "This defense allows more than the player's norm — lean over.";
    color = "var(--pos)";
  } else if (relative < -deadband) {
    verdict = "Tough spot";
    detail = "This defense allows less than the player's norm — lean under.";
    color = "var(--neg)";
  } else {
    verdict = "Neutral matchup";
    detail = "This defense allows about what the player averages.";
    color = "var(--ink-2)";
  }

  const bg =
    color === "var(--ink-2)"
      ? "var(--bg-2)"
      : `color-mix(in oklab, ${color} 10%, transparent)`;

  return (
    <div
      style={{
        marginTop: 12,
        padding: "10px 12px",
        borderRadius: 6,
        background: bg,
        borderLeft: `3px solid ${color}`,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color }}>{verdict}</div>
      <div className="dim" style={{ fontSize: 10.5, marginTop: 2 }}>
        {detail}
      </div>
    </div>
  );
}

/** Signed delta tag (allowed − player avg). */
function DeltaTag({ delta }: { delta: number }) {
  const positive = delta >= 0;
  const color = positive ? "var(--pos)" : "var(--neg)";
  return (
    <span className="mono tnum" style={{ color, fontWeight: 500 }}>
      ({positive ? "+" : ""}
      {delta.toFixed(1)} vs norm)
    </span>
  );
}

/** Comparison table: player avg vs defense-allowed + rank, per split. */
function ComparisonTable({
  games,
  cohorts,
  playerName,
}: {
  games: { week: number; value: number | null; is_home: boolean }[];
  cohorts: Record<string, CohortAllowed> | null;
  playerName: string;
}) {
  return (
    <div style={{ marginTop: 20 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 10 }}>
        By split
      </div>
      <table
        className="mono tnum"
        style={{ width: "100%", fontSize: 11.5, borderCollapse: "collapse" }}
      >
        <thead>
          <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
            <th style={{ padding: "6px 12px 6px 0" }}>Split</th>
            <th style={{ padding: "6px 12px 6px 0", textAlign: "right" }}>
              {playerName}
            </th>
            <th style={{ padding: "6px 12px 6px 0", textAlign: "right" }}>
              Def allowed
            </th>
            <th style={{ padding: "6px 0", textAlign: "right" }}>Def rank</th>
          </tr>
        </thead>
        <tbody>
          {PLAYER_SPLITS.map((s, i) => {
            const first = i === 0;
            const border = first ? "none" : "1px solid var(--line-soft)";
            if (s.pending) {
              return (
                <tr key={s.key} style={{ borderTop: border }}>
                  <td style={{ padding: "7px 12px 7px 0", color: "var(--ink-2)" }}>
                    {s.label}
                  </td>
                  <td colSpan={3} style={{ padding: "7px 0", textAlign: "right" }}>
                    <PendingChip>pending</PendingChip>
                  </td>
                </tr>
              );
            }
            const pAvg = playerAvgForSplit(games, s.key);
            const c = cohorts?.[s.key];
            const allowed = c?.avg_allowed ?? null;
            const rank = c?.rank_against_position ?? null;
            return (
              <tr key={s.key} style={{ borderTop: border }}>
                <td style={{ padding: "7px 12px 7px 0", color: "var(--ink-2)" }}>
                  {s.label}
                </td>
                <td style={{ padding: "7px 12px 7px 0", textAlign: "right", color: "var(--ink)" }}>
                  {pAvg != null ? pAvg.toFixed(1) : "—"}
                </td>
                <td style={{ padding: "7px 12px 7px 0", textAlign: "right", color: "var(--info)" }}>
                  {allowed != null ? allowed.toFixed(1) : "—"}
                </td>
                <td style={{ padding: "7px 0", textAlign: "right", color: "var(--ink-3)" }}>
                  {rank != null ? ordinal(rank) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * C2 — per-game bar chart. Bars = player's stat per game (B1); solid
 * reference line = team's split-average allowed (B3, moves with split);
 * book line pending (odds).
 */
function PlayerBarChartCard({
  playerId,
  playerName,
  statLabel,
  statKey,
  statType,
  team,
  split,
}: {
  playerId: string;
  playerName: string;
  statLabel: string;
  statKey: string;
  statType: string;
  team: string;
  split: PlayerSplitKey;
}) {
  const history = usePlayerHistory(playerId, { stat: statKey });
  const defense = useDefenseAllowed(team, { stat_type: statType });

  const bars = (history.data?.items ?? []).map((g) => ({
    label: String(g.week),
    value: g.value ?? null,
  }));

  // Team-allowed average for the selected split (client-side pick).
  const cohorts = defense.data?.cohorts as
    | Record<string, { avg_allowed?: number | null }>
    | null
    | undefined;
  // Pending splits (vs_winning/losing/top10) have no cohort → null line.
  const refValue = cohorts?.[split]?.avg_allowed ?? null;

  const loading = history.isLoading || defense.isLoading;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          {playerName} · {statLabel} · per game
          {history.data?.season != null && (
            <span className="mono dim2" style={{ marginLeft: 8, fontSize: 9 }}>
              {history.data.season}-{history.data.season + 1} season
            </span>
          )}
        </div>
        <div
          className="mono dim2"
          style={{
            fontSize: 10,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <span style={{ color: "var(--info)" }}>
            — {team} allowed ({split})
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ opacity: 0.5 }}>- -</span>
            <PendingChip>book line pending</PendingChip>
          </span>
        </div>
      </div>

      {loading ? (
        <div className="dim" style={{ padding: 20 }}>Loading…</div>
      ) : (
        <BarChart
          bars={bars}
          referenceValue={refValue}
          referenceLabel="allowed"
        />
      )}
    </div>
  );
}

/** Searchable player combobox: text input + client-filtered result list. */
function PlayerCombobox({
  players,
  selected,
  onSelect,
}: {
  players: { player_id: string; player_name: string; position: string; team: string }[];
  selected: { player_id: string; player_name: string; position: string; team: string } | null;
  onSelect: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = query.trim()
    ? players
        .filter((p) => p.player_name.toLowerCase().includes(query.toLowerCase()))
        .slice(0, 50)
    : players.slice(0, 50);

  return (
    <div style={{ position: "relative" }}>
      <input
        type="text"
        value={open ? query : (selected ? `${selected.player_name} (${selected.position} · ${selected.team})` : query)}
        placeholder="Search player…"
        onFocus={() => { setOpen(true); setQuery(""); }}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        style={selectStyle}
      />
      {open && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 20,
            marginTop: 2,
            maxHeight: 240,
            overflowY: "auto",
            background: "var(--bg-2)",
            border: "1px solid var(--line-soft)",
            borderRadius: 5,
            boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
          }}
        >
          {filtered.length === 0 && (
            <div className="dim mono" style={{ padding: "8px 10px", fontSize: 11 }}>No match.</div>
          )}
          {filtered.map((p) => (
            <button
              key={p.player_id}
              type="button"
              onClick={() => { onSelect(p.player_id); setOpen(false); setQuery(""); }}
              style={{
                display: "flex",
                width: "100%",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "6px 10px",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                font: "inherit",
                color: "var(--ink)",
                textAlign: "left",
              }}
              onMouseDown={(e) => e.preventDefault()}
            >
              <span style={{ fontSize: 12 }}>{p.player_name}</span>
              <span className="mono dim2" style={{ fontSize: 10 }}>{p.position} · {p.team}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Pending (non-clickable) split pill — participates in highlight mode. */
function PendingSplitPill({ label }: { label: string }) {
  const highlight = usePendingHighlight();
  return (
    <span
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 10px",
        fontSize: 11,
        color: "var(--ink-4)",
        border: "1px solid var(--line-soft)",
        borderRadius: 4,
        cursor: "not-allowed",
        ...highlight,
      }}
      title="Split pending backend support"
    >
      {label}
      <span style={{ fontSize: 8 }}>⏳</span>
    </span>
  );
}

const selectStyle: React.CSSProperties = {
  background: "var(--bg-1)",
  color: "var(--ink)",
  border: "1px solid var(--line-soft)",
  borderRadius: 5,
  padding: "6px 10px",
  fontSize: 12,
  fontFamily: "var(--f-sans)",
  width: "100%",
};

const PLAYER_TEAMS = [
  { value: "", label: "Select defense…" },
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

function StatRowLabelStatus({ status }: { status: FieldStatus | undefined }) {
  if (!status) return null;
  if (status === "pending") return <PendingField placeholder="" />;
  return (
    <BlockedField
      blocker={status.blocker}
      roadmap={status.roadmap}
      placeholder=""
    />
  );
}

function CompareCell({
  value,
  status,
}: {
  value: number | string | null | undefined;
  status: FieldStatus | undefined;
}) {
  if (value != null && value !== "") {
    return <>{typeof value === "number" ? formatNumericValue(value) : value}</>;
  }
  // Row is field_status-scaffolded; just show em dash (label already shows badge).
  if (status) return <span className="dim2">—</span>;
  return <span className="dim2">—</span>;
}

function formatNumericValue(v: number): string {
  // Elo-style ratings render as whole numbers, other numerics get one decimal.
  if (Math.abs(v) > 100) return v.toFixed(0);
  return v.toFixed(1);
}

type MatchupEdge = {
  metric: MatchupMetric;
  offTeam: string;
  defTeam: string;
  edge: number; // defRank - offRank; positive = offense favored
};

/**
 * Find the biggest collision (largest rank-differential edge) for one
 * direction. Returns null if no ranked metric available.
 */
function biggestCollision(
  metrics: MatchupMetric[],
  offCohort: Record<string, number>,
  defCohort: Record<string, number>,
  offTeam: string,
  defTeam: string,
): MatchupEdge | null {
  let best: MatchupEdge | null = null;
  for (const m of metrics) {
    const offRank = offCohort[`rank_${m.off}`];
    const defRank = defCohort[`rank_${m.def}`];
    if (offRank == null || defRank == null) continue;
    const edge = defRank - offRank; // high = strong offense vs weak defense
    if (best == null || Math.abs(edge) > Math.abs(best.edge)) {
      best = { metric: m, offTeam, defTeam, edge };
    }
  }
  return best;
}

/** Rank-differential magnitude → descriptor word. */
function edgeDescriptor(edge: number): string {
  const mag = Math.abs(edge);
  if (mag >= 15) return "Big edge";
  if (mag >= 7) return "Edge";
  if (mag >= 3) return "Slight edge";
  return "Even";
}

/**
 * Auto-generated narrative banner. Computes the biggest matchup collision
 * in each direction from rank differentials and describes them in plain
 * language. Updates with the selected cohort.
 */
function NarrativeBanner({
  teamA,
  teamB,
  cohortA,
  cohortB,
}: {
  teamA: string;
  teamB: string;
  cohortA: Record<string, number>;
  cohortB: Record<string, number>;
}) {
  const aCollision = biggestCollision(
    MATCHUP_METRICS,
    cohortA,
    cohortB,
    teamA,
    teamB,
  );
  const bCollision = biggestCollision(
    MATCHUP_METRICS,
    cohortB,
    cohortA,
    teamB,
    teamA,
  );

  if (!aCollision && !bCollision) return null;

  const describe = (c: MatchupEdge): string => {
    const descriptor = edgeDescriptor(c.edge);
    if (descriptor === "Even") {
      return `${c.offTeam}'s ${c.metric.title.toLowerCase()} vs ${c.defTeam}'s defense — even matchup`;
    }
    const favored = c.edge > 0 ? c.offTeam : c.defTeam;
    return `${c.offTeam}'s ${c.metric.title.toLowerCase()} vs ${c.defTeam}'s defense — ${descriptor} ${favored}`;
  };

  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        fontSize: 11.5,
        lineHeight: 1.5,
        color: "var(--ink-2)",
      }}
    >
      <span style={{ color: "var(--info)", fontSize: 14, flexShrink: 0 }}>
        ⌖
      </span>
      <div>
        {aCollision && (
          <div>
            <span className="dim">Biggest collision: </span>
            {describe(aCollision)}.
          </div>
        )}
        {bCollision && (
          <div style={{ marginTop: 2 }}>
            <span className="dim">Other side: </span>
            {describe(bCollision)}.
          </div>
        )}
      </div>
    </div>
  );
}
