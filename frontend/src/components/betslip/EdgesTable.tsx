import { useEdges } from "../../api/hooks";
import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";
import { TeamMark } from "../primitives/TeamMark";
import { useBetSlip } from "../../context/BetSlipContext";
import type { BetLeg } from "../../context/BetSlipContext";
import { ErrorCard } from "../../components/error/ErrorCard";

type EdgeRowShape = {
  game_id: string;
  game_date?: string | null;
  away_team: string;
  home_team: string;
  market_type: string;
  side: string;
  model_value?: number | null;
  market_value?: number | null;
  point_edge?: number | null;
  cover_prob?: number | null;
  ev: number;
  edge_strength: string;
  kelly_frac?: number | null;
  kelly_stake?: number | null;
};

export function EdgesTable() {
  const { data, isLoading, error, refetch } = useEdges();
  const { legs, add } = useBetSlip();

  const legIds = new Set(legs.map((l) => l.id));

  const listStatus = data?._meta?.field_status?.items as FieldStatus | undefined;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Available Edges
        </div>
        {data && (
          <div className="mono dim2" style={{ fontSize: 10 }}>
            {data.season} · Week {data.week} · min EV {(data.min_ev ?? 0).toFixed(2)}
          </div>
        )}
      </div>

      {isLoading && <div className="dim">Loading…</div>}

      {error && (
        <ErrorCard
          error={error}
          onRetry={() => refetch()}
          title="Couldn't load games"
        />
      )}


      {data && (data.items ?? []).length === 0 && (
        <ListEmptyState status={listStatus} />
      )}

      {data && (data.items ?? []).length > 0 && (
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
              <th style={{ padding: "8px 12px 8px 0" }}>Matchup</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Market</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Side</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>EV</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Strength</th>
              <th style={{ padding: "8px 0" }}></th>
            </tr>
          </thead>
          <tbody>
            {(data.items ?? []).map((edge, i) => {
              const legId = buildLegId(edge);
              const alreadyAdded = legIds.has(legId);
              return (
                <tr
                  key={i}
                  style={{ borderTop: "1px solid var(--line-soft)" }}
                >
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                      }}
                    >
                      <TeamMark abbr={edge.away_team} />
                      <span className="dim">@</span>
                      <TeamMark abbr={edge.home_team} />
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    {edge.market_type}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>{edge.side}</td>
                  <td
                    style={{
                      padding: "10px 12px 10px 0",
                      textAlign: "right",
                      color:
                        edge.ev >= 0.05
                          ? "var(--pos)"
                          : edge.ev >= 0.02
                            ? "var(--warn)"
                            : "var(--ink-2)",
                    }}
                  >
                    {(edge.ev * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <EdgeStrengthPill strength={edge.edge_strength} />
                  </td>
                  <td style={{ padding: "10px 0" }}>
                    <AddButton
                      disabled={alreadyAdded}
                      onClick={() => add(edgeToLeg(edge))}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ListEmptyState({ status }: { status: FieldStatus | undefined }) {
  return (
    <div
      style={{
        padding: 24,
        textAlign: "center",
      }}
    >
      <div
        className="mono dim"
        style={{ fontSize: 12, marginBottom: 8 }}
      >
        No edges available.
      </div>
      {status === "pending" && (
        <div style={{ display: "inline-block" }}>
          <PendingField />
        </div>
      )}
      {status && status !== "pending" && (
        <div style={{ display: "inline-block" }}>
          <BlockedField blocker={status.blocker} roadmap={status.roadmap} />
        </div>
      )}
      {status && status !== "pending" && (
        <div
          className="mono dim2"
          style={{ fontSize: 11, marginTop: 8 }}
        >
          {status.blocker === "no_odds_available"
            ? "Run `gridiron ingest dk-odds` to refresh."
            : status.blocker === "no_champion_manifest"
              ? "Run `gridiron evaluate select-model --write-manifest`."
              : "See the operational checklist."}
        </div>
      )}
    </div>
  );
}

function EdgeStrengthPill({ strength }: { strength: string }) {
  const color =
    strength === "strong"
      ? "var(--pos)"
      : strength === "moderate"
        ? "var(--warn)"
        : "var(--ink-3)";

  return (
    <span
      className="mono upper"
      style={{
        fontSize: 9,
        color,
        padding: "2px 6px",
        border: `1px solid ${color}`,
        borderRadius: 3,
      }}
    >
      {strength}
    </span>
  );
}

function AddButton({
  disabled,
  onClick,
}: {
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      style={{
        background: disabled ? "var(--bg-2)" : "var(--pos)",
        color: disabled ? "var(--ink-4)" : "var(--bg)",
        border: "none",
        borderRadius: 4,
        padding: "4px 10px",
        fontSize: 11,
        fontWeight: 600,
        fontFamily: "var(--f-sans)",
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      {disabled ? "Added" : "Add"}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Edge → BetLeg conversion
// ---------------------------------------------------------------------------

function buildLegId(edge: EdgeRowShape): string {
  return `${edge.game_id}__${edge.market_type}__${edge.side}`;
}

function edgeToLeg(edge: EdgeRowShape): BetLeg {
  // Derive American odds from the edge's market_value.
  // For moneyline, market_value IS the American odds.
  // For spread and total, market_value is the line; edge doesn't include the odds
  // explicitly. Default to -110 as a stand-in.
  const isMoneyline = edge.market_type === "moneyline";
  const odds = isMoneyline && edge.market_value != null
    ? Math.round(edge.market_value)
    : -110;

  const isSpreadOrTotal = edge.market_type === "spread" || edge.market_type === "total";
  const line = isSpreadOrTotal && edge.market_value != null ? edge.market_value : undefined;

  return {
    id: buildLegId(edge),
    gameId: edge.game_id,
    market: edge.market_type as "moneyline" | "spread" | "total",
    side: edge.side as "home" | "away" | "over" | "under",
    odds,
    line,
    awayTeam: edge.away_team,
    homeTeam: edge.home_team,
  };
}
