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

      {/* Recent results */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Recent Results
        </div>
        <RecentResultsStrip results={data.recent_results} />
      </div>

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
        <ScaffoldCard
          title="Situational Splits"
          status={fieldStatus?.cohort_splits as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

// function ProfileCell({
//   label,
//   value,
// }: {
//   label: string;
//   value: React.ReactNode;
// }) {
//   return (
//     <div style={{ minWidth: 100 }}>
//       <div className="upper dim2" style={{ fontSize: 10, marginBottom: 6 }}>
//         {label}
//       </div>
//       <div className="mono tnum" style={{ fontSize: 14 }}>
//         {value}
//       </div>
//     </div>
//   );
// }

// function InlineFieldStatus({ status }: { status: FieldStatus | undefined }) {
//   if (!status) return <span className="mono tnum dim2">—</span>;
//   if (status === "pending") return <PendingField />;
//   return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
// }

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
