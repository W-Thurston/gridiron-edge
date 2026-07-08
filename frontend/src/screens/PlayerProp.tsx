import { useComparePlayer, useProp, useGame } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";
import { useTeamByAbbr } from "../api/team_metadata_hook";
import { useBetSlip } from "../context/BetSlipContext";
import { formatStatType } from "../utils/props";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { DistributionChart } from "../components/primitives/DistributionChart";
import { WhyLink } from "../components/primitives/WhyLink";

export function PlayerProp() {
  const { route, navigate } = useNav();
  const { legs, add } = useBetSlip();
  const propId = route.params.propId ?? null;

  const propResult = useProp(propId);
  const compareResult = useComparePlayer(propId);
  const gameResult = useGame(propResult.data?.game_id ?? null);

  if (!propId) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">No prop selected.</div>
      </div>
    );
  }

  const backNav = (
    <div>
      <button
        type="button"
        onClick={() => navigate("/players")}
        className="dim mono"
        style={{
          background: "transparent",
          border: "none",
          padding: 0,
          cursor: "pointer",
          font: "inherit",
          color: "var(--ink-3)",
          fontSize: 12,
        }}
      >
        ← Players
      </button>
    </div>
  );

  if (propResult.isLoading || compareResult.isLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">Loading…</div>
        </div>
      </div>
    );
  }

  if (propResult.error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <ErrorCard
          error={propResult.error as Error}
          onRetry={() => propResult.refetch()}
        />
      </div>
    );
  }

  const prop = propResult.data;
  const compare = compareResult.data;
  const game = gameResult.data ?? null;

  if (!prop) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">No prop data available.</div>
        </div>
      </div>
    );
  }

  // Bet slip integration
  const legId = `player-prop-${prop.prop_id}`;
  const isOnSlip = legs.some((l) => l.id === legId);
  const handleAddSlip = () => {
    if (isOnSlip) return;
    add({
      id: legId,
      gameId: prop.prop_id,
      market: "prop" as never,
      side: "over" as "home" | "away" | "over" | "under",
      odds: -110,
      awayTeam: prop.team,
      homeTeam: prop.team,
    });
  };

  const propStatus = prop._meta?.field_status;
  const compareStatus = compare?._meta?.field_status;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {backNav}

      {/* Player hero band */}
      <PlayerHero
        prop={{
          prop_id: prop.prop_id,
          player_name: prop.player_name,
          position: prop.position,
          team: prop.team,
          season: prop.season,
          stat_type: prop.stat_type,
          projection: prop.projection,
        }}
        onAddSlip={handleAddSlip}
        isOnSlip={isOnSlip}
        gameOpponent={game && prop.team ? getOpponentFromGameId(prop.game_id, prop.team) : null}
        gameDayOfWeek={game?.day_of_week ?? null}
        confidenceTier={prop.line_context?.confidence_tier ?? null}
      />


      {/* Projection distribution */}
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
            Projection distribution
          </div>
          <div
            className="mono dim2"
            style={{ fontSize: 10 }}
          >
            90% credible band
          </div>
        </div>
        <DistributionChart
          mean={prop.projection?.predicted_mean}
          std={prop.projection?.predicted_std}
          lo={prop.projection?.lo_90}
          hi={prop.projection?.hi_90}
        />
      </div>

      {/* Situational splits */}
      <SituationalSplitsCard situationalSplits={prop.situational_splits} />

      {/* Player vs Defense */}
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
            Player vs Defense
          </div>
          <WhyLink
            dot
            tone="info"
            subject={{ kind: "prop_defense", propId: prop.prop_id }}
          />
        </div>
        {compare && (compare.stats ?? []).length > 0 ? (
          <table
            className="mono tnum"
            style={{
              width: "100%",
              fontSize: 12,
              borderCollapse: "collapse",
            }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    padding: "6px 12px 6px 0",
                    textAlign: "left",
                    fontSize: 10,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontWeight: 400,
                    color: "var(--ink-3)",
                  }}
                >
                  Stat
                </th>
                <th
                  style={{
                    padding: "6px 12px 6px 0",
                    textAlign: "right",
                    fontSize: 10,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontWeight: 400,
                    color: "var(--ink-3)",
                  }}
                >
                  Projection
                </th>
                <th
                  style={{
                    padding: "6px 0",
                    textAlign: "right",
                    fontSize: 10,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontWeight: 400,
                    color: "var(--ink-3)",
                  }}
                >
                  Defense
                </th>
              </tr>
            </thead>
            <tbody>
              {(compare.stats ?? []).map((row) => (
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
                    style={{
                      padding: "10px 12px 10px 0",
                      textAlign: "right",
                    }}
                  >
                    <CompareCell value={row.projection_value} />
                  </td>
                  <td style={{ padding: "10px 0", textAlign: "right" }}>
                    <CompareCell
                      value={row.defense_value}
                      status={
                        compareStatus?.[row.key] as FieldStatus | undefined
                      }
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="dim mono">No comparison data available.</div>
        )}
      </div>

      {/* Blocked/pending section placeholders */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 16,
        }}
      >
        <ComingSoonCard
          title="Historical vs Opponent"
          status={propStatus?.historical_vs_opponent as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Recent Form"
          status={propStatus?.recent_form as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Injury Status"
          status={propStatus?.injury_status as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Prop Reasoning"
          status={propStatus?.prop_reasoning as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Multi-Book Shopping"
          status={propStatus?.multi_book_shopping as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

/**
 * Player hero band. Team-colored vertical gradient, TeamMark on left,
 * breadcrumb above serif player name, prop summary callout on right.
 * Same design language as TeamsScreen's TeamHeroBand.
 */
function PlayerHero({
  prop,
  onAddSlip,
  isOnSlip,
  gameOpponent,
  gameDayOfWeek,
  confidenceTier,
}: {
  prop: {
    prop_id: string;
    player_name: string;
    position: string;
    team: string;
    season?: string | null;
    stat_type: string;
    projection?: {
      predicted_mean?: number | null;
      lo_90?: number | null;
      hi_90?: number | null;
    } | null;
  };
  onAddSlip: () => void;
  isOnSlip: boolean;
  gameOpponent?: string | null;
  gameDayOfWeek?: string | null;
  confidenceTier?: string | null;
}) {
  const teamMetadata = useTeamByAbbr(prop.team);
  const primaryColor = teamMetadata?.primary_color;

  // Vertical gradient from team color top → var(--bg-1) bottom
  const background = primaryColor
    ? `linear-gradient(180deg, color-mix(in oklab, ${primaryColor} 30%, var(--bg)) 0%, var(--bg-1) 100%)`
    : "var(--bg-1)";

  // Format breadcrumb
  const seasonPart = prop.season ? formatSeason(prop.season) : null;
  const breadcrumbParts = [
    prop.position,
    prop.team,
    seasonPart ? `${seasonPart} Season` : null,
  ].filter((p): p is string => p != null);
  const breadcrumb = breadcrumbParts.join(" · ");

  const statLabel = formatStatType(prop.stat_type);
  const modelMean = prop.projection?.predicted_mean;
  const lo = prop.projection?.lo_90;
  const hi = prop.projection?.hi_90;

  return (
    <div
      className="hm-card"
      style={{
        padding: "24px 28px",
        background,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
        }}
      >
        {/* Left: Team mark + player info */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            flex: 1,
            minWidth: 0,
          }}
        >
          <TeamMark abbr={prop.team} size={56} />
          <div style={{ flex: 1, minWidth: 0 }}>
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
            <div
              style={{
                fontFamily: "var(--f-serif)",
                fontSize: 30,
                fontWeight: 400,
                color: "var(--ink)",
                lineHeight: 1.1,
              }}
            >
              {prop.player_name}
            </div>
          </div>
        </div>

        {/* Right: Prop summary card + slip button */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Prop summary card */}
          <div
            style={{
              background: "var(--bg-1)",
              border: "1px solid var(--line-soft)",
              borderLeft: "3px solid var(--pos)",
              borderRadius: 6,
              padding: "12px 20px",
              display: "flex",
              flexDirection: "column",
              gap: 4,
              minWidth: 240,
            }}
          >
            {/* Header row */}
            <div
              className="mono upper"
              style={{
                fontSize: 9.5,
                letterSpacing: "0.1em",
                color: "var(--ink-3)",
              }}
            >
              {statLabel}
              {gameDayOfWeek && gameOpponent && (
                <span style={{ color: "var(--ink-4)" }}>
                  {" · "}
                  {gameDayOfWeek.slice(0, 3)} vs {gameOpponent}
                </span>
              )}
            </div>

            {/* Line (pending — big em-dash) */}
            <div
              style={{
                fontSize: 24,
                fontWeight: 600,
                color: "var(--ink-3)",
                fontFamily: "var(--f-mono)",
                lineHeight: 1,
                marginTop: 4,
              }}
            >
              —{" "}
              <span
                className="mono"
                style={{
                  fontSize: 12,
                  color: "var(--ink-4)",
                  fontWeight: 400,
                }}
              >
                (line pending)
              </span>
            </div>

            {/* Model + range on same line */}
            <div
              style={{
                display: "flex",
                gap: 16,
                marginTop: 4,
                fontSize: 11,
                color: "var(--ink-3)",
              }}
              className="mono"
            >
              {modelMean != null && (
                <span>
                  Model mean{" "}
                  <span style={{ color: "var(--ink)", fontWeight: 500 }}>
                    {modelMean.toFixed(1)}
                  </span>
                </span>
              )}
              {lo != null && hi != null && (
                <span>
                  Range{" "}
                  <span style={{ color: "var(--ink)", fontWeight: 500 }}>
                    {lo.toFixed(0)}–{hi.toFixed(0)}
                  </span>
                </span>
              )}
            </div>
            {/* Confidence tier tag + EV */}
            <div
              style={{
                marginTop: 6,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              {confidenceTier ? (
                <ConfidenceTierPill tier={confidenceTier} />
              ) : (
                <span
                  className="mono"
                  style={{
                    fontSize: 10,
                    color: "var(--ink-4)",
                    padding: "2px 6px",
                    border: "1px solid var(--line-soft)",
                    borderRadius: 3,
                  }}
                >
                  Confidence pending
                </span>
              )}
              <span
                className="mono"
                style={{
                  fontSize: 10,
                  color: "var(--ink-4)",
                }}
              >
                EV pending
              </span>
            </div>
          </div>

          {/* Slip button — outside the summary card */}
          <button
            onClick={onAddSlip}
            type="button"
            disabled={isOnSlip}
            style={{
              padding: "10px 18px",
              background: isOnSlip ? "var(--bg-3)" : "var(--pos)",
              color: isOnSlip ? "var(--ink-4)" : "var(--bg)",
              border: "none",
              borderRadius: 4,
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "var(--f-sans)",
              cursor: isOnSlip ? "default" : "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {isOnSlip ? "✓ On slip" : "+ Bet slip"}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Format season string. "2025-2026" → "2025".
 */
function formatSeason(season: string): string {
  const parts = season.split("-");
  return parts[0] ?? season;
}


function CompareCell({
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

function ComingSoonCard({
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
        <ComingSoonStatus status={status} />
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

function ComingSoonStatus({ status }: { status: FieldStatus | undefined }) {
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

/**
 * Parse a game_id and return the opponent's team abbreviation.
 *
 * game_id format: "2025_08_KAN_BAL" (season_week_away_home)
 * If playerTeam matches one of the two teams, return the other.
 * Otherwise return null.
 */
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

type CohortInfo = {
  key: string;
  label: string;
};

const SPLIT_COHORTS: CohortInfo[] = [
  { key: "season", label: "Season" },
  { key: "l4", label: "Last 4 games" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
  { key: "favored", label: "Favored" },
  { key: "underdog", label: "Underdog" },
  { key: "indoor", label: "Indoor" },
  { key: "outdoor", label: "Outdoor" },
];

/**
 * Situational Splits card for player prop. Renders 8 cohorts from Step 5
 * situational_splits data. Each row: cohort label + mean value + sample size.
 *
 * When situational_splits is null (pending), shows empty state with pending
 * indicator. When individual cohorts are missing, shows em-dash rows.
 */
function SituationalSplitsCard({
  situationalSplits,
}: {
  situationalSplits: { [key: string]: unknown } | null | undefined;
}) {
  const hasData = situationalSplits != null;

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
        {!hasData && <PendingField placeholder="" />}
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
          Splits data not yet available for this prop.
        </div>
      ) : (
        <div style={{ display: "grid", gap: 4 }}>
          {SPLIT_COHORTS.map((cohort, i) => (
            <SplitRow
              key={cohort.key}
              label={cohort.label}
              data={
                situationalSplits[cohort.key] as
                  | { sample_size?: number; mean_value?: number }
                  | undefined
              }
              first={i === 0}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Single cohort row in Situational Splits.
 */
function SplitRow({
  label,
  data,
  first,
}: {
  label: string;
  data: { sample_size?: number; mean_value?: number } | undefined;
  first: boolean;
}) {
  const meanValue = data?.mean_value;
  const sampleSize = data?.sample_size;

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
          color: meanValue != null ? "var(--ink)" : "var(--ink-4)",
          fontWeight: 500,
        }}
      >
        {meanValue != null ? (
          <>
            {meanValue.toFixed(1)}
            {sampleSize != null && (
              <span
                className="dim2"
                style={{
                  fontSize: 10,
                  fontWeight: 400,
                  marginLeft: 6,
                }}
              >
                avg · {sampleSize} games
              </span>
            )}
          </>
        ) : (
          "—"
        )}
      </span>
    </div>
  );
}
