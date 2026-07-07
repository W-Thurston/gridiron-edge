import { useState } from "react";

import { useTeamProfile, useTeamRankings } from "../api/hooks";
import { TeamMark } from "../components/primitives/TeamMark";
import { PendingField } from "../components/field-status/PendingField";
import { BlockedField } from "../components/field-status/BlockedField";
import type { FieldStatus } from "../components/field-status/types";
import { RecentResultsStrip } from "../components/teams/RecentResultsStrip";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";
import { Pill } from "../components/primitives/Pill";
import { RatingChart } from "../components/primitives/RatingChart";
import { useProjections } from "../api/hooks";

/**
 * Consolidated split-view Teams screen. Left column shows rankings;
 * right column shows the currently-selected team's profile. URL param
 * `?team=X` drives selection.
 *
 * If no team param, auto-selects #1 ranked team silently (no URL update).
 * Row click updates URL param without navigation.
 *
 * Old routes `/teams/:abbr` redirect to `/teams?team=abbr` via Router.
 */
export function TeamsScreen() {
  const { route, navigate } = useNav();
  const rankingsResult = useTeamRankings();

  // URL param drives selection; fall back to #1 team when empty
  const teamParam = route.params.team ?? null;
  const rankings = rankingsResult.data?.items ?? [];
  const defaultTeam = rankings[0]?.abbr ?? null;
  const selectedAbbr = teamParam ?? defaultTeam;

  const profileResult = useTeamProfile(selectedAbbr);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "2fr 3fr", gap: 16 }}>
      {/* Left column: Rankings */}
      <RankingsColumn
        rankings={rankings}
        isLoading={rankingsResult.isLoading}
        error={rankingsResult.error as Error | null}
        onRetry={() => rankingsResult.refetch()}
        selectedAbbr={selectedAbbr}
        onSelectTeam={(abbr) => navigate("/teams", { team: abbr })}
      />

      {/* Right column: Team profile */}
      <ProfileColumn
        abbr={selectedAbbr}
        result={profileResult}
      />
    </div>
  );
}

/**
 * Left column: rankings table. Row click updates selection via
 * onSelectTeam callback (URL param, not navigation).
 *
 * Enhancements in Substep 2a:
 * - Trend column wired to real backend data
 * - Signed colored pill for trend badge
 * - Hover state on rows
 * - Tighter row height for 32-team fit
 */
type RankingsTab = "overall" | "offense" | "defense" | "ats" | "net";

const TABS: { key: RankingsTab; label: string; blocked: boolean }[] = [
  { key: "overall", label: "Overall", blocked: false },
  { key: "offense", label: "Offense", blocked: true },
  { key: "defense", label: "Defense", blocked: true },
  { key: "ats", label: "ATS", blocked: true },
  { key: "net", label: "Net Rating", blocked: true },
];

function RankingsColumn({
  rankings,
  isLoading,
  error,
  onRetry,
  selectedAbbr,
  onSelectTeam,
}: {
  rankings: Array<{
    abbr: string;
    name: string;
    rating?: number | null;
    rank?: number | null;
    record?: { wins: number; losses: number; ties: number } | null;
    trend?: number | null;
  }>;
  isLoading: boolean;
  error: Error | null;
  onRetry: () => void;
  selectedAbbr: string | null;
  onSelectTeam: (abbr: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<RankingsTab>("overall");
  const activeTabInfo = TABS.find((t) => t.key === activeTab);
  const isBlocked = activeTabInfo?.blocked ?? false;

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
          Power Rankings
        </div>
        {rankings.length > 0 && (
          <span className="mono dim2" style={{ fontSize: 10 }}>
            {rankings.length} teams
          </span>
        )}
      </div>

      {/* Tab strip */}
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 12,
          paddingBottom: 12,
          borderBottom: "1px solid var(--line-soft)",
        }}
      >
        {TABS.map((tab) => (
          <Pill
            key={tab.key}
            active={activeTab === tab.key}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </Pill>
        ))}
      </div>

      {error && (
        <ErrorCard
          error={error}
          onRetry={onRetry}
          title="Couldn't load rankings"
        />
      )}

      {!error && isBlocked && (
        <BlockedTabState tab={activeTab} />
      )}

      {!error && !isBlocked && (
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
              <th
                style={{
                  padding: "6px 12px 6px 0",
                  textAlign: "right",
                  fontSize: 10,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontWeight: 400,
                }}
              >
                #
              </th>
              <th
                style={{
                  padding: "6px 12px 6px 0",
                  fontSize: 10,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontWeight: 400,
                }}
              >
                Team
              </th>
              <th
                style={{
                  padding: "6px 12px 6px 0",
                  textAlign: "right",
                  fontSize: 10,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontWeight: 400,
                }}
              >
                Rating
              </th>
              <th
                style={{
                  padding: "6px 12px 6px 0",
                  textAlign: "right",
                  fontSize: 10,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontWeight: 400,
                }}
              >
                Record
              </th>
              <th
                style={{
                    padding: "6px 8px 6px 0",
                    textAlign: "right",
                    fontSize: 10,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontWeight: 400,
                }}
                >
                Trend
                </th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={5}>
                  <div className="dim" style={{ padding: "20px 0" }}>
                    Loading…
                  </div>
                </td>
              </tr>
            )}
            {!isLoading && rankings.length === 0 && (
              <tr>
                <td colSpan={5}>
                  <div className="dim mono" style={{ fontSize: 12, padding: "20px 0" }}>
                    No team ratings found.
                  </div>
                </td>
              </tr>
            )}
            {!isLoading &&
              rankings.map((team) => {
                const isSelected = team.abbr === selectedAbbr;
                return (
                  <RankingRow
                    key={team.abbr}
                    team={team}
                    isSelected={isSelected}
                    onClick={() => onSelectTeam(team.abbr)}
                  />
                );
              })}
          </tbody>
        </table>
      )}
    </div>
  );
}

/**
 * Individual row in the rankings table with hover state.
 */
function RankingRow({
  team,
  isSelected,
  onClick,
}: {
  team: {
    abbr: string;
    name: string;
    rating?: number | null;
    rank?: number | null;
    record?: { wins: number; losses: number; ties: number } | null;
    trend?: number | null;
  };
  isSelected: boolean;
  onClick: () => void;
}) {
  const [isHover, setIsHover] = useState(false);

  const bg = isSelected
    ? "color-mix(in oklab, var(--pos) 8%, transparent)"
    : isHover
      ? "color-mix(in oklab, var(--ink) 3%, transparent)"
      : "transparent";

  return (
    <tr
      onClick={onClick}
      onMouseEnter={() => setIsHover(true)}
      onMouseLeave={() => setIsHover(false)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      tabIndex={0}
      role="button"
      aria-label={`Select ${team.name}`}
      aria-current={isSelected ? "true" : undefined}
      style={{
        borderTop: "1px solid var(--line-soft)",
        cursor: "pointer",
        background: bg,
        transition: "background 90ms ease",
      }}
    >
      <td
        style={{
          padding: "6px 12px 6px 0",
          textAlign: "right",
          color: "var(--ink-3)",
        }}
      >
        {team.rank ?? "—"}
      </td>
      <td style={{ padding: "6px 12px 6px 0" }}>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <TeamMark abbr={team.abbr} size={18} />
          <span
            style={{
              color: isSelected ? "var(--ink)" : "var(--ink-2)",
              fontWeight: isSelected ? 500 : 400,
            }}
          >
            {team.name}
          </span>
        </span>
      </td>
      <td
        style={{
          padding: "6px 12px 6px 0",
          textAlign: "right",
        }}
      >
        {team.rating?.toFixed(0) ?? "—"}
      </td>
      <td
        style={{
          padding: "6px 12px 6px 0",
          textAlign: "right",
          color: "var(--ink-2)",
        }}
      >
        {team.record
          ? `${team.record.wins}-${team.record.losses}${
              team.record.ties > 0 ? `-${team.record.ties}` : ""
            }`
          : "—"}
      </td>
      <td style={{ padding: "6px 8px 6px 0", textAlign: "right" }}>
        <TrendBadge trend={team.trend} />
      </td>
    </tr>
  );
}

/**
 * Signed colored pill for team trend. Positive = green, negative = red,
 * zero or missing = dim.
 */
function TrendBadge({ trend }: { trend: number | null | undefined }) {
  if (trend == null) {
    return <span className="mono dim2">—</span>;
  }

  const isPositive = trend > 0;
  const isNegative = trend < 0;
  const color = isPositive
    ? "var(--pos)"
    : isNegative
      ? "var(--neg)"
      : "var(--ink-3)";
  const bg = isPositive
    ? "color-mix(in oklab, var(--pos) 14%, transparent)"
    : isNegative
      ? "color-mix(in oklab, var(--neg) 14%, transparent)"
      : "var(--bg-2)";

  const sign = trend > 0 ? "+" : "";
  const formatted = `${sign}${trend.toFixed(1)}`;

  return (
    <span
      className="mono tnum"
      style={{
        display: "inline-block",
        padding: "2px 8px",
        fontSize: 10,
        color,
        background: bg,
        borderRadius: 3,
        fontWeight: 600,
        minWidth: 40,
      }}
    >
      {formatted}
    </span>
  );
}

/**
 * Right column: team profile. Preserves existing TeamProfile UI as-is.
 * Tier 3 substeps replace this section-by-section with real composition.
 */
function ProfileColumn({
  abbr,
  result,
}: {
  abbr: string | null;
  result: ReturnType<typeof useTeamProfile>;
}) {
  if (!abbr) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">Select a team from the rankings.</div>
      </div>
    );
  }

  if (result.isLoading) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">Loading team profile…</div>
      </div>
    );
  }

  if (result.error) {
    return (
      <ErrorCard
        error={result.error as Error}
        onRetry={() => result.refetch()}
        title={`Couldn't load ${abbr}`}
      />
    );
  }

  if (!result.data) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">No team data available.</div>
      </div>
    );
  }

  const data = result.data;
  const fieldStatus = data._meta?.field_status;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Team hero band */}
      <TeamHeroBand data={data} />

      {/* Rating chart */}
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Power rating · season trend
        </div>
        <RatingChart history={data.rating_history} />
      </div>

      {/* Cohort splits */}
      <CohortSplitsCard cohortSplits={data.cohort_splits} />

      {/* Recent results */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Recent Results
        </div>
        <RecentResultsStrip results={data.recent_results} />
      </div>

      {/* Schedule difficulty (blocked placeholder) */}
      <ScheduleDifficultyPlaceholder />

      {/* Postseason outlook */}
      <PostseasonOutlookCard teamAbbr={data.abbr} />

      {/* Scaffolded cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <ScaffoldCard
          title="Top Players"
          status={fieldStatus?.top_players as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

function ScaffoldCard({
  title,
  status,
}: {
  title: string;
  status: FieldStatus | undefined;
}) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        className="upper dim"
        style={{
          fontSize: 10,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>{title}</span>
        {status === "pending" && <PendingField placeholder="" />}
        {status && status !== "pending" && (
          <BlockedField
            blocker={status.blocker}
            roadmap={status.roadmap}
            placeholder=""
          />
        )}
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Not yet available
      </div>
    </div>
  );
}

/**
 * Shown when user selects a rankings tab that isn't backed by data yet.
 * Explains what's needed and why. Uses same visual language as
 * ComingSoonCard but scoped to tab content area.
 */
function BlockedTabState({ tab }: { tab: RankingsTab }) {
  const info: Record<RankingsTab, { title: string; reason: string }> = {
    overall: { title: "Overall Rankings", reason: "" }, // never rendered
    offense: {
      title: "Offense Rankings",
      reason: "Requires offensive rating decomposition (backend work; ROADMAP §9.7).",
    },
    defense: {
      title: "Defense Rankings",
      reason: "Requires defensive rating decomposition (backend work; ROADMAP §9.7).",
    },
    ats: {
      title: "ATS Rankings",
      reason: "Requires cumulative ATS record enrichment (backend work; ROADMAP §9.7).",
    },
    net: {
      title: "Net Rating",
      reason: "Requires off/def rating decomposition (backend work; ROADMAP §9.7).",
    },
  };

  const { title, reason } = info[tab];

  return (
    <div
      style={{
        padding: 32,
        textAlign: "center",
        color: "var(--ink-3)",
      }}
    >
      <div
        style={{
          fontSize: 12,
          marginBottom: 8,
          color: "var(--ink-2)",
          fontWeight: 500,
        }}
      >
        {title}
      </div>
      <div style={{ fontSize: 11.5, lineHeight: 1.5 }}>
        {reason}
      </div>
    </div>
  );
}

/**
 * Team hero band. Team-colored gradient background, TeamMark on left,
 * title block on right with breadcrumb + big serif italic team name +
 * inline hero stats.
 */
function TeamHeroBand({
  data,
}: {
  data: {
    abbr: string;
    name: string;
    city?: string | null;
    conference?: string | null;
    division?: string | null;
    primary_color?: string | null;
    season?: string | null;
    as_of_week?: number | null;
    rating?: number | null;
    rank?: number | null;
    record?: { wins: number; losses: number; ties: number } | null;
  };
}) {
  const primaryColor = data.primary_color;
  // Gradient from team color at ~25% to darker fade
  const background = primaryColor
    ? `linear-gradient(180deg, color-mix(in oklab, ${primaryColor} 30%, var(--bg)) 0%, var(--bg-1) 100%)`
    : "var(--bg-1)";

  // Format breadcrumb
  const divExpanded = expandDivisionLetter(data.division ?? "");
  const confDiv = data.conference && data.division
    ? `${data.conference} ${divExpanded}`
    : null;
  const rankPart = data.rank != null ? `#${data.rank} Power` : null;
  const seasonPart = data.season ? formatSeason(data.season) : null;
  const weekPart = data.as_of_week != null ? `Through Wk ${data.as_of_week}` : null;
  const breadcrumbParts = [confDiv, rankPart, seasonPart, weekPart].filter(
    (p): p is string => p != null,
  );
  const breadcrumb = breadcrumbParts.join(" · ");

  // Team name split
  const nameWithoutCity = stripCityPrefix(data.name, data.city ?? undefined);
  const cityPart = data.city ?? "";

  // Format record stat
  const recordText = data.record
    ? `${data.record.wins}-${data.record.losses}${
        data.record.ties > 0 ? `-${data.record.ties}` : ""
      }`
    : "—";

  return (
    <div
      className="hm-card"
      style={{
        padding: "24px 28px",
        background,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        {/* TeamMark on left */}
        <TeamMark abbr={data.abbr} size={56} />

        {/* Title block on right */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Breadcrumb above title */}
          <div
            className="mono upper"
            style={{
              fontSize: 10.5,
              color: "var(--ink-3)",
              letterSpacing: "0.08em",
              marginBottom: 4,
            }}
          >
            {breadcrumb || "—"}
          </div>

          {/* Big serif italic team name */}
          <div
            style={{
              fontFamily: "var(--f-serif)",
              fontSize: 30,
              fontWeight: 400,
              color: "var(--ink)",
              lineHeight: 1.1,
              marginBottom: 8,
            }}
          >
            {cityPart} <span style={{ fontStyle: "italic" }}>{nameWithoutCity}</span>
          </div>

          {/* Inline hero stats */}
          <div
            className="mono"
            style={{
              fontSize: 12,
              color: "var(--ink-3)",
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            <span>
              Record{" "}
              <span style={{ color: "var(--ink-2)", fontWeight: 500 }}>
                {recordText}
              </span>
            </span>
            <span>
              Rank{" "}
              <span style={{ color: "var(--ink-2)", fontWeight: 500 }}>
                {data.rank != null ? `#${data.rank}` : "—"}
              </span>
            </span>
            <span>
              Rating{" "}
              <span style={{ color: "var(--ink-2)", fontWeight: 500 }}>
                {data.rating?.toFixed(0) ?? "—"}
              </span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Strip city prefix from team name (matches GameDetail helper).
 * "Kansas City Chiefs" + city "Kansas City" → "Chiefs"
 */
function stripCityPrefix(
  name: string | undefined | null,
  city: string | undefined,
): string {
  if (!name) return "—";
  if (!city) return name;
  const prefix = `${city} `;
  if (name.startsWith(prefix)) {
    return name.slice(prefix.length);
  }
  return name;
}

/**
 * Expand division letter to full name.
 * N → North, S → South, E → East, W → West
 */
function expandDivisionLetter(letter: string): string {
  const map: Record<string, string> = {
    N: "North",
    S: "South",
    E: "East",
    W: "West",
  };
  return map[letter.toUpperCase()] ?? letter;
}

/**
 * Format season string. "2025-2026" → "2025".
 */
function formatSeason(season: string): string {
  const parts = season.split("-");
  return parts[0] ?? season;
}

type CohortKey = "season" | "l4" | "home" | "away";

const COHORT_TABS: { key: CohortKey; label: string }[] = [
  { key: "season", label: "Season" },
  { key: "l4", label: "Last 4" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
];

const SPLIT_METRICS: {
  key: string;
  label: string;
  fmt: (v: number) => string;
}[] = [
  { key: "off_epa_per_play", label: "Off. EPA/play", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_pass_epa", label: "Pass EPA/play", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_rush_epa", label: "Rush EPA/play", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "def_epa_per_play", label: "Def. EPA/play", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "def_rush_epa", label: "Def. Rush EPA", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_third_down_pct", label: "3rd-down conv.", fmt: (v) => (v * 100).toFixed(1) + "%" },
  { key: "off_redzone_td_pct", label: "Red-zone TD %", fmt: (v) => (v * 100).toFixed(1) + "%" },
  { key: "turnover_diff", label: "Turnover diff", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(1) },
];

/**
 * Situational Splits card for team profile. Renders 8 metrics for the
 * selected cohort. Uses Pill primitive for cohort switching.
 */
function CohortSplitsCard({
  cohortSplits,
}: {
  cohortSplits: { [key: string]: unknown } | null | undefined;
}) {
  const [cohort, setCohort] = useState<CohortKey>("season");

  const data = cohortSplits?.[cohort] as
    | Record<string, number>
    | undefined;
  const hasData = data != null;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Situational Splits
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {COHORT_TABS.map((tab) => (
            <Pill
              key={tab.key}
              active={cohort === tab.key}
              onClick={() => setCohort(tab.key)}
            >
              {tab.label}
            </Pill>
          ))}
        </div>
      </div>

      {!hasData ? (
        <div
          style={{
            padding: 20,
            textAlign: "center",
            color: "var(--ink-4)",
            fontSize: 12,
          }}
        >
          No data available for {COHORT_TABS.find((t) => t.key === cohort)?.label}.
        </div>
      ) : (
        <div style={{ display: "grid", gap: 4 }}>
          {SPLIT_METRICS.map((metric, i) => (
            <SplitMetricRow
              key={metric.key}
              label={metric.label}
              value={data[metric.key]}
              fmt={metric.fmt}
              first={i === 0}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Single metric row: 2-column with label on left, value on right.
 */
function SplitMetricRow({
  label,
  value,
  fmt,
  first,
}: {
  label: string;
  value: number | null | undefined;
  fmt: (v: number) => string;
  first: boolean;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 12,
        alignItems: "center",
        padding: "8px 0",
        borderTop: first ? "none" : "1px solid var(--line-soft)",
        fontSize: 12,
      }}
    >
      <span className="dim mono" style={{ fontSize: 10.5, letterSpacing: "0.04em", textTransform: "uppercase" }}>
        {label}
      </span>
      <span
        className="mono tnum"
        style={{
          color: "var(--ink)",
          fontWeight: 500,
        }}
      >
        {value != null ? fmt(value) : "—"}
      </span>
    </div>
  );
}

/**
 * Postseason outlook card for team profile. Filters /projections
 * response client-side to find the current team, then renders 5
 * probability rows for postseason milestones.
 */
function PostseasonOutlookCard({ teamAbbr }: { teamAbbr: string }) {
  const { data, isLoading, error } = useProjections();

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Postseason Outlook
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Postseason Outlook
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load projections.
        </div>
      </div>
    );
  }

  const teamProjection = data?.items?.find((row) => row.abbr === teamAbbr);

  if (!teamProjection) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Postseason Outlook
        </div>
        <div
          style={{
            padding: 20,
            textAlign: "center",
            color: "var(--ink-4)",
            fontSize: 12,
          }}
        >
          No projection data available.
        </div>
      </div>
    );
  }

  const rows: { label: string; value: number | null | undefined }[] = [
    { label: "Make Playoffs", value: teamProjection.make_playoffs },
    { label: "Reach Divisional", value: teamProjection.reach_div },
    { label: "Reach Conf. Championship", value: teamProjection.reach_conf },
    { label: "Reach Super Bowl", value: teamProjection.reach_sb },
    { label: "Win Super Bowl", value: teamProjection.win_sb },
  ];

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Postseason Outlook
      </div>

      <div style={{ display: "grid", gap: 4 }}>
        {rows.map((row, i) => (
          <PostseasonRow
            key={row.label}
            label={row.label}
            value={row.value}
            first={i === 0}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Single row in postseason outlook: label on left, percentage on right.
 */
function PostseasonRow({
  label,
  value,
  first,
}: {
  label: string;
  value: number | null | undefined;
  first: boolean;
}) {
  const pct = value != null ? Math.round(value * 100) : null;
  const formatted = pct != null ? `${pct}%` : "—";

  // Color by probability strength
  const color =
    pct == null
      ? "var(--ink-3)"
      : pct >= 75
        ? "var(--pos)"
        : pct >= 50
          ? "var(--info)"
          : pct >= 20
            ? "var(--ink)"
            : "var(--ink-3)";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 12,
        alignItems: "center",
        padding: "8px 0",
        borderTop: first ? "none" : "1px solid var(--line-soft)",
        fontSize: 12,
      }}
    >
      <span
        className="dim mono"
        style={{
          fontSize: 10.5,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
      <span
        className="mono tnum"
        style={{
          color,
          fontWeight: 500,
        }}
      >
        {formatted}
      </span>
    </div>
  );
}

/**
 * Placeholder for the schedule difficulty section.
 * Blocked on backend work per ROADMAP §9.7 — requires upcoming_games
 * enrichment with per-opponent difficulty index.
 */
function ScheduleDifficultyPlaceholder() {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        className="upper dim"
        style={{
          fontSize: 10,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>Schedule Difficulty</span>
        <BlockedField
          blocker="schedule_difficulty"
          roadmap="§9.7"
          placeholder=""
        />
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Upcoming opponents + difficulty index not yet available.
      </div>
    </div>
  );
}
