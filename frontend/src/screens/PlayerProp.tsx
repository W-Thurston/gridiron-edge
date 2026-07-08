import { useComparePlayer, useProp } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";
import { useTeamByAbbr } from "../api/team_metadata_hook";
import { useBetSlip } from "../context/BetSlipContext";
import { formatStatType } from "../utils/props";

export function PlayerProp() {
  const { route, navigate } = useNav();
  const { legs, add } = useBetSlip();
  const propId = route.params.propId ?? null;

  const propResult = useProp(propId);
  const compareResult = useComparePlayer(propId);

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
      />

      {/* Distribution chart placeholder — Tier 3a replaces */}
      <DistributionPlaceholder prop={prop} />

      {/* Situational Splits placeholder — Tier 3b replaces */}
      <SectionPlaceholder title="Situational Splits" />

      {/* Player vs Defense (existing table, polished in 3c) */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Player vs Defense
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
              <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
                <th style={{ padding: "8px 12px 8px 0" }}>Stat</th>
                <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                  Projection
                </th>
                <th style={{ padding: "8px 0", textAlign: "right" }}>Defense</th>
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

      {/* Scaffolded cards from PropDetail (moved into own layout in 4a) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
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

        {/* Right: Prop summary callout */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: 4,
            minWidth: 180,
          }}
        >
          <div
            className="upper dim"
            style={{
              fontSize: 9.5,
              letterSpacing: "0.1em",
              color: "var(--ink-4)",
            }}
          >
            {statLabel}
          </div>

          {/* Line (pending — big em-dash) */}
          <div
            style={{
              fontSize: 18,
              fontWeight: 600,
              color: "var(--ink-3)",
              fontFamily: "var(--f-mono)",
            }}
          >
            —
          </div>
          <div
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-4)",
              marginTop: -2,
              marginBottom: 4,
            }}
          >
            line pending
          </div>

          {/* Model + range */}
          {modelMean != null && (
            <div
              className="mono"
              style={{ fontSize: 11, color: "var(--ink-3)" }}
            >
              Model{" "}
              <span style={{ color: "var(--ink)", fontWeight: 500 }}>
                {modelMean.toFixed(1)}
              </span>
            </div>
          )}
          {lo != null && hi != null && (
            <div
              className="mono"
              style={{ fontSize: 10.5, color: "var(--ink-4)" }}
            >
              Range {lo.toFixed(0)}–{hi.toFixed(0)}
            </div>
          )}

          {/* Slip button */}
          <button
            onClick={onAddSlip}
            type="button"
            disabled={isOnSlip}
            style={{
              padding: "6px 14px",
              background: isOnSlip ? "var(--bg-3)" : "var(--pos)",
              color: isOnSlip ? "var(--ink-4)" : "var(--bg)",
              border: "none",
              borderRadius: 4,
              fontSize: 12,
              fontWeight: 600,
              fontFamily: "var(--f-sans)",
              cursor: isOnSlip ? "default" : "pointer",
              marginTop: 6,
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

/**
 * Distribution chart placeholder. Renders existing projection cells
 * until Tier 3a replaces with DistributionChart primitive.
 */
function DistributionPlaceholder({
  prop,
}: {
  prop: {
    projection?: {
      predicted_mean?: number | null;
      predicted_std?: number | null;
      lo_90?: number | null;
      hi_90?: number | null;
    } | null;
  };
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Projection
      </div>
      {prop.projection ? (
        <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
          <ProjectionCell
            label="Predicted Mean"
            value={prop.projection.predicted_mean?.toFixed(1) ?? "—"}
          />
          <ProjectionCell
            label="Std (Uncertainty)"
            value={prop.projection.predicted_std?.toFixed(1) ?? "—"}
          />
          <ProjectionCell
            label="10th %ile"
            value={prop.projection.lo_90?.toFixed(0) ?? "—"}
          />
          <ProjectionCell
            label="90th %ile"
            value={prop.projection.hi_90?.toFixed(0) ?? "—"}
          />
        </div>
      ) : (
        <div className="dim mono">No projection available.</div>
      )}
    </div>
  );
}

/**
 * Titled empty card for sections in-progress this workstream.
 */
function SectionPlaceholder({ title }: { title: string }) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div className="upper dim" style={{ fontSize: 10 }}>
        {title}
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Coming in Tier 3
      </div>
    </div>
  );
}

function ProjectionCell({
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
