import { useTeamProfile, useTeamRankings } from "../api/hooks";
import { TeamMark } from "../components/primitives/TeamMark";
import { PendingField } from "../components/field-status/PendingField";
import { BlockedField } from "../components/field-status/BlockedField";
import type { FieldStatus } from "../components/field-status/types";
import { Spark } from "../components/primitives/Spark";
import { RecentResultsStrip } from "../components/teams/RecentResultsStrip";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";

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
 */
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
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Power Rankings
      </div>

      {isLoading && <div className="dim">Loading…</div>}

      {error && (
        <ErrorCard
            error={error}
            onRetry={onRetry}
            title="Couldn't load rankings"
        />
        )}

      {!isLoading && !error && rankings.length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No team ratings found.
        </div>
      )}

      {!isLoading && !error && rankings.length > 0 && (
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
              <th style={{ padding: "6px 12px 6px 0", textAlign: "right" }}>#</th>
              <th style={{ padding: "6px 12px 6px 0" }}>Team</th>
              <th style={{ padding: "6px 12px 6px 0", textAlign: "right" }}>
                Rating
              </th>
              <th style={{ padding: "6px 12px 6px 0", textAlign: "right" }}>
                Record
              </th>
              <th style={{ padding: "6px 0" }}>Trend</th>
            </tr>
          </thead>
          <tbody>
            {rankings.map((team) => {
              const isSelected = team.abbr === selectedAbbr;
              return (
                <tr
                  key={team.abbr}
                  onClick={() => onSelectTeam(team.abbr)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelectTeam(team.abbr);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Select ${team.name}`}
                  aria-current={isSelected ? "true" : undefined}
                  style={{
                    borderTop: "1px solid var(--line-soft)",
                    cursor: "pointer",
                    background: isSelected
                      ? "color-mix(in oklab, var(--pos) 6%, transparent)"
                      : "transparent",
                  }}
                >
                  <td
                    style={{
                      padding: "8px 12px 8px 0",
                      textAlign: "right",
                      color: "var(--ink-3)",
                    }}
                  >
                    {team.rank ?? "—"}
                  </td>
                  <td style={{ padding: "8px 12px 8px 0" }}>
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
                  <td style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                    {team.rating?.toFixed(0) ?? "—"}
                  </td>
                  <td
                    style={{
                      padding: "8px 12px 8px 0",
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
                  <td style={{ padding: "8px 0" }}>—</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
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
      {/* Team profile — existing structure. Tier 3 replaces section by section. */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Team Profile — {data.season} through Week {data.as_of_week}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 20,
          }}
        >
          <TeamMark abbr={data.abbr} />
          <span>{data.name}</span>
        </div>

        <div style={{ display: "flex", gap: 32, marginTop: 20, flexWrap: "wrap" }}>
          <ProfileCell label="Rating" value={data.rating?.toFixed(0) ?? "—"} />
          <ProfileCell label="Rank" value={data.rank?.toString() ?? "—"} />
          <ProfileCell
            label="Record"
            value={
              data.record
                ? `${data.record.wins}-${data.record.losses}${
                    data.record.ties > 0 ? `-${data.record.ties}` : ""
                  }`
                : "—"
            }
          />
          <ProfileCell
            label="Off Rating"
            value={<InlineFieldStatus status={fieldStatus?.off_rating as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Def Rating"
            value={<InlineFieldStatus status={fieldStatus?.def_rating as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Trend"
            value={data.trend?.toFixed(1) ?? "—"}
          />
        </div>
      </div>

      {/* Rating history */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Rating Trajectory ({data.season})
        </div>
        <Spark
            data={data.rating_history?.map((p) => p.rating) ?? []}
            width={480}
            height={60}
        />
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

function ProfileCell({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div style={{ minWidth: 100 }}>
      <div className="upper dim2" style={{ fontSize: 10, marginBottom: 6 }}>
        {label}
      </div>
      <div className="mono tnum" style={{ fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}

function InlineFieldStatus({ status }: { status: FieldStatus | undefined }) {
  if (!status) return <span className="mono tnum dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
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
