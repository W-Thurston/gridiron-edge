import { useEffect, useState } from "react";
import { useCompareTeams, useTeamProfile, usePropsList,useComparePlayer } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/primitives/TeamMark";
import { Pill } from "../components/primitives/Pill";
import { TeamPicker } from "../components/compare/TeamPicker";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";
import { formatStatType } from "../utils/props";
import { DistributionChart } from "../components/primitives/DistributionChart";

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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
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

/**
 * Collapsible summary card — the non-cohort stat table (rating/rank/
 * record/percentiles). Collapsed by default; secondary to the matchup
 * sections.
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
        <table
          className="mono tnum"
          style={{
            width: "100%",
            fontSize: 12,
            borderCollapse: "collapse",
            marginTop: 12,
          }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>Stat</th>
              <th
                style={{
                  padding: "8px 12px 8px 0",
                  textAlign: "right",
                  fontWeight: 500,
                  color: "var(--ink)",
                }}
              >
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    justifyContent: "flex-end",
                  }}
                >
                  <TeamMark abbr={data.team_a} />
                  {data.team_a}
                </span>
              </th>
              <th
                style={{
                  padding: "8px 0",
                  textAlign: "right",
                  fontWeight: 500,
                  color: "var(--ink)",
                }}
              >
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    justifyContent: "flex-end",
                  }}
                >
                  <TeamMark abbr={data.team_b} />
                  {data.team_b}
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {(data.stats ?? []).map((row) => (
              <StatRowDisplay
                key={row.key}
                row={row}
                status={
                  data._meta?.field_status?.[row.key] as
                    | FieldStatus
                    | undefined
                }
              />
            ))}
          </tbody>
        </table>
      )}
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
  label: string;
  fmt: (v: number) => string;
};

const epaFmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(3);
const pctFmt = (v: number) => (v * 100).toFixed(1) + "%";

const MATCHUP_METRICS: MatchupMetric[] = [
  { off: "off_epa_per_play", def: "def_epa_per_play", label: "EPA / play", fmt: epaFmt },
  { off: "off_pass_epa", def: "def_pass_epa", label: "Pass EPA", fmt: epaFmt },
  { off: "off_rush_epa", def: "def_rush_epa", label: "Rush EPA", fmt: epaFmt },
  { off: "off_third_down_pct", def: "def_third_down_pct", label: "3rd-down %", fmt: pctFmt },
  { off: "off_redzone_td_pct", def: "def_redzone_td_pct", label: "Red-zone TD %", fmt: pctFmt },
];

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
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600 }}>{title}</div>
        <div className="dim mono" style={{ fontSize: 10.5 }}>
          {subtitle}
        </div>
      </div>

      {/* Column headers */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          gap: 12,
          fontSize: 9.5,
          color: "var(--ink-4)",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        <span style={{ textAlign: "left" }}>{offTeam} offense</span>
        <span style={{ textAlign: "center" }}>matchup</span>
        <span style={{ textAlign: "right" }}>{defTeam} defense</span>
      </div>

      <div style={{ display: "grid", gap: 4 }}>
        {MATCHUP_METRICS.map((m, i) => {
          const offVal = offCohort[m.off];
          const defVal = defCohort[m.def];
          return (
            <div
              key={m.off}
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto 1fr",
                gap: 12,
                alignItems: "center",
                padding: "8px 0",
                borderTop: i === 0 ? "none" : "1px solid var(--line-soft)",
                fontSize: 12,
              }}
            >
              <span
                className="mono tnum"
                style={{ textAlign: "left", color: "var(--ink)" }}
              >
                {offVal != null ? m.fmt(offVal) : "—"}
              </span>
              <span
                className="dim"
                style={{
                  textAlign: "center",
                  fontSize: 10,
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  minWidth: 100,
                }}
              >
                {m.label}
              </span>
              <span
                className="mono tnum"
                style={{ textAlign: "right", color: "var(--ink-2)" }}
              >
                {defVal != null ? m.fmt(defVal) : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Neutral section: turnover_diff head-to-head. */
function EvenFooting({
  cohortA,
  cohortB,
}: {
  cohortA: Record<string, number>;
  cohortB: Record<string, number>;
}) {
  const aVal = cohortA["turnover_diff"];
  const bVal = cohortB["turnover_diff"];
  const fmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(3);

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600 }}>Even footing</div>
        <div className="dim mono" style={{ fontSize: 10.5 }}>
          Neutral metrics
        </div>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          gap: 12,
          alignItems: "center",
          padding: "8px 0",
          fontSize: 12,
        }}
      >
        <span
          className="mono tnum"
          style={{ textAlign: "left", color: "var(--ink)" }}
        >
          {aVal != null ? fmt(aVal) : "—"}
        </span>
        <span
          className="dim"
          style={{
            textAlign: "center",
            fontSize: 10,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            minWidth: 100,
          }}
        >
          Turnover diff
        </span>
        <span
          className="mono tnum"
          style={{ textAlign: "right", color: "var(--ink-2)" }}
        >
          {bVal != null ? fmt(bVal) : "—"}
        </span>
      </div>
    </div>
  );
}

/**
 * Player vs Defense comparison mode. Placeholder — Tier 3 builds the
 * player + defense pickers, DistributionChart, and stat rows.
 */
function PlayerCompareMode() {
  const { route, navigate } = useNav();
  const selectedPropId = route.params.prop_id ?? "";

  const propsList = usePropsList({});
  const props = propsList.data?.items ?? [];

  const setPropId = (propId: string) => {
    const params: Record<string, string> = { mode: "player" };
    if (propId) params.prop_id = propId;
    navigate("/compare", params);
  };

  const selectedProp = props.find((p) => p.prop_id === selectedPropId) ?? null;
  const opponent = selectedProp
    ? getOpponentFromGameId(selectedProp.game_id, selectedProp.team)
    : null;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Player vs Defense
      </div>

      {/* Prop picker */}
      <div style={{ marginBottom: 16 }}>
        <span className="upper dim2" style={{ fontSize: 9 }}>
          Prop
        </span>
        <select
          value={selectedPropId}
          onChange={(e) => setPropId(e.target.value)}
          style={{
            display: "block",
            marginTop: 4,
            background: "var(--bg-1)",
            color: "var(--ink)",
            border: "1px solid var(--line-soft)",
            borderRadius: 5,
            padding: "6px 10px",
            fontSize: 12,
            fontFamily: "var(--f-sans)",
            minWidth: 320,
          }}
        >
          <option value="">Select a prop…</option>
          {props.map((p) => (
            <option key={p.prop_id} value={p.prop_id}>
              {p.player_name} · {formatStatType(p.stat_type)} ({p.team})
            </option>
          ))}
        </select>
      </div>

      {/* Derived matchup header */}
      {selectedProp && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "12px 0",
            borderTop: "1px solid var(--line-soft)",
            marginBottom: 8,
          }}
        >
          <TeamMark abbr={selectedProp.team} size={28} />
          <div>
            <div style={{ fontWeight: 500 }}>{selectedProp.player_name}</div>
            <div className="mono dim" style={{ fontSize: 11 }}>
              {formatStatType(selectedProp.stat_type)} · {selectedProp.position}
            </div>
          </div>
          <span
            className="serif"
            style={{
              fontSize: 18,
              fontStyle: "italic",
              color: "var(--ink-3)",
              margin: "0 8px",
            }}
          >
            vs
          </span>
          {opponent ? (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <TeamMark abbr={opponent} size={28} />
              <div className="mono dim" style={{ fontSize: 11 }}>
                {opponent} defense
              </div>
            </div>
          ) : (
            <span className="dim mono" style={{ fontSize: 11 }}>
              defense
            </span>
          )}
        </div>
      )}

      {/* Comparison content */}
      {!selectedProp && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          Select a prop to compare against its opponent's defense.
        </div>
      )}
      {selectedProp && <PlayerDefenseComparison propId={selectedProp.prop_id} />}
    </div>
  );
}

/**
 * Fetches /compare/player/{prop_id} and renders the projection
 * distribution + player-vs-defense stat rows.
 */
function PlayerDefenseComparison({ propId }: { propId: string }) {
  const { data, isLoading, error, refetch } = useComparePlayer(propId);

  if (isLoading) {
    return <div className="dim" style={{ marginTop: 8 }}>Loading…</div>;
  }
  if (error) {
    return (
      <ErrorCard
        error={error as Error}
        onRetry={() => refetch()}
        title="Couldn't load comparison"
      />
    );
  }
  if (!data) {
    return (
      <div className="dim mono" style={{ fontSize: 12 }}>
        No comparison data available.
      </div>
    );
  }

  const stats = data.stats ?? [];
  const fieldStatus = data._meta?.field_status;

  // Extract distribution params from projection rows
  const getProj = (key: string) =>
    stats.find((s) => s.key === key)?.projection_value ?? null;
  const mean = getProj("mean");
  const std = getProj("std");
  const lo = getProj("lo_90");
  const hi = getProj("hi_90");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Distribution chart */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 12,
          }}
        >
          <div className="upper dim" style={{ fontSize: 10 }}>
            Projection distribution
          </div>
          <div className="mono dim2" style={{ fontSize: 10 }}>
            90% credible band
          </div>
        </div>
        <DistributionChart
          mean={typeof mean === "number" ? mean : null}
          std={typeof std === "number" ? std : null}
          lo={typeof lo === "number" ? lo : null}
          hi={typeof hi === "number" ? hi : null}
        />
      </div>

      {/* Player vs Defense stat rows */}
      <div>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Projection vs Defense
        </div>
        <table
          className="mono tnum"
          style={{
            width: "100%",
            fontSize: 12,
            borderCollapse: "collapse",
          }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>Stat</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Projection
              </th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>Defense</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((row) => (
              <tr
                key={row.key}
                style={{ borderTop: "1px solid var(--line-soft)" }}
              >
                <td
                  style={{
                    padding: "10px 12px 10px 0",
                    color: "var(--ink-2)",
                  }}
                >
                  {row.label}
                </td>
                <td
                  style={{ padding: "10px 12px 10px 0", textAlign: "right" }}
                >
                  <PlayerCompareCell value={row.projection_value} />
                </td>
                <td style={{ padding: "10px 0", textAlign: "right" }}>
                  <PlayerCompareCell
                    value={row.defense_value}
                    status={fieldStatus?.[row.key] as FieldStatus | undefined}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PlayerCompareCell({
  value,
  status,
}: {
  value: number | string | null | undefined;
  status?: FieldStatus | undefined;
}) {
  if (value != null && value !== "") {
    return <>{typeof value === "number" ? value.toFixed(1) : value}</>;
  }
  if (!status) return <span className="dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

/** Parse game_id into opponent abbrev given the player's team. */
function getOpponentFromGameId(
  gameId: string,
  playerTeam: string,
): string | null {
  const parts = gameId.split("_");
  if (parts.length < 4) return null;
  const [, , away, home] = parts;
  if (playerTeam === home) return away;
  if (playerTeam === away) return home;
  return null;
}

type StatRow = {
  key: string;
  label: string;
  unit?: string | null;
  team_a_value?: number | string | null;
  team_b_value?: number | string | null;
};

function StatRowDisplay({
  row,
  status,
}: {
  row: StatRow;
  status: FieldStatus | undefined;
}) {
  return (
    <tr style={{ borderTop: "1px solid var(--line-soft)" }}>
      <td
        style={{
          padding: "10px 12px 10px 0",
          color: "var(--ink-2)",
        }}
      >
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          {row.label}
          <StatRowLabelStatus status={status} />
        </span>
      </td>
      <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
        <CompareCell value={row.team_a_value} status={status} />
      </td>
      <td style={{ padding: "10px 0", textAlign: "right" }}>
        <CompareCell value={row.team_b_value} status={status} />
      </td>
    </tr>
  );
}

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
      return `${c.offTeam}'s ${c.metric.label.toLowerCase()} vs ${c.defTeam}'s defense — even matchup`;
    }
    const favored = c.edge > 0 ? c.offTeam : c.defTeam;
    return `${c.offTeam}'s ${c.metric.label.toLowerCase()} vs ${c.defTeam}'s defense — ${descriptor} ${favored}`;
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
