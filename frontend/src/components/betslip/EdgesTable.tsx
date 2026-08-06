import { Fragment, useState } from "react";
import { useEdges } from "../../api/hooks";
import { EdgeResultStatus } from "../field-status/EdgeResultStatus";
import { TeamMark } from "../primitives/TeamMark";
import { useBetSlip } from "../../context/BetSlipContext";
import { useAppState } from "../../context/AppStateContext";
import {
  edgeOfferKey,
  filterEdgesBySportsbook,
  groupEdgeOffers,
  sportsbookDisplayName,
} from "../../utils/sportsbookPreferences";
import { buildGameBetLegId, createGameBetLeg } from "../../utils/betLegs";
import { ErrorCard } from "../../components/error/ErrorCard";

export function EdgesTable({
  bankroll,
  kellyMultiplier,
}: {
  bankroll: number | null;
  kellyMultiplier: number;
}) {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    () => new Set(),
  );
  const edgeParams =
    bankroll == null
      ? {
          kelly_multiplier:
            kellyMultiplier,
        }
      : {
          bankroll,
          kelly_multiplier:
            kellyMultiplier,
        };

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useEdges(edgeParams);

  const { legs, add } = useBetSlip();
  const { state } = useAppState();
  const items = filterEdgesBySportsbook(data?.items ?? [], state);
  const groups = groupEdgeOffers(items);

  const legIds = new Set(legs.map((leg) => leg.id));

  const toggleGroup = (groupId: string) => {
    setExpandedGroups((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

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
        <div
          id="available-edges-heading"
          className="upper dim"
          style={{ fontSize: 10 }}
        >
          Available Edges
        </div>
        {data && (
          <div className="mono dim2" style={{ fontSize: 10 }}>
            {data.season} · Week {data.week} · min EV {(data.min_ev ?? 0).toFixed(2)}
          </div>
        )}
      </div>
      <div
        className="mono dim2"
        style={{
          marginBottom: 14,
          fontSize: 10,
          lineHeight: 1.5,
        }}
      >
        Stage a model edge, then adjust
        current odds and proposed stake in
        the Bet Slip.
      </div>

      {isLoading && (
        <div
          className="dim"
          role="status"
          aria-live="polite"
        >
          Loading edges…
        </div>
      )}

      {error && (
        <ErrorCard
          error={error}
          onRetry={() => refetch()}
          title="Couldn't load games"
        />
      )}


      {data && items.length === 0 && (
        <EdgeResultStatus diagnostics={data.diagnostics} />
      )}

      {data && groups.length > 0 && (
      <div
        className="betslip-edges-scroll"
        role="region"
        aria-labelledby="available-edges-heading"
        tabIndex={0}
      >
        <table
          className="betslip-edges-table mono tnum"
          style={{
            fontSize: 12,
          }}
        >
          <caption className="betslip-edges-caption">
            Available model edges that can
            be staged in the Bet Slip
          </caption>
          <thead>
            <tr
              style={{
                color: "var(--ink-3)",
                textAlign: "left",
              }}
            >
              <th
                scope="col"
                style={{
                  padding:
                    "8px 12px 8px 0",
                }}
              >
                Matchup
              </th>

              <th
                scope="col"
                style={{ padding: "8px 12px 8px 0" }}
              >
                Sportsbook
              </th>

              <th
                scope="col"
                style={{
                  padding:
                    "8px 12px 8px 0",
                }}
              >
                Market
              </th>

              <th
                scope="col"
                style={{
                  padding:
                    "8px 12px 8px 0",
                }}
              >
                Side
              </th>

              <th
                scope="col"
                style={{ padding: "8px 12px 8px 0" }}
              >
                Market Context
              </th>

              <th
                scope="col"
                style={{ padding: "8px 12px 8px 0" }}
              >
                Odds
              </th>

              <th
                scope="col"
                style={{
                  padding:
                    "8px 12px 8px 0",
                  textAlign: "right",
                }}
              >
                EV
              </th>

              <th
                scope="col"
                style={{
                  padding:
                    "8px 12px 8px 0",
                }}
              >
                Strength
              </th>

              <th
                scope="col"
                aria-label="Bet Slip action"
                style={{
                  padding: "8px 0",
                }}
              />
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => {
              const isExpanded = expandedGroups.has(group.id);
              const alternativesId = `available-edge-offers-${group.id.replaceAll(":", "-")}`;

              const renderOfferRow = (
                edge: typeof group.best,
                isAlternative: boolean,
                alternativeIndex?: number,
              ) => {
                const market =
                  edge.market_type as
                    | "moneyline"
                    | "spread"
                    | "total";
                const side =
                  edge.side as
                    | "home"
                    | "away"
                    | "over"
                    | "under";
                const line =
                  market === "spread" || market === "total"
                    ? edge.market_value ?? null
                    : null;
                const legId = buildGameBetLegId({
                  gameId: edge.game_id,
                  market,
                  side,
                  line,
                  sportsbook: edge.sportsbook ?? null,
                });
                const alreadyAdded = legIds.has(legId);
                const sportsbook = edge.sportsbook
                  ? sportsbookDisplayName(edge.sportsbook)
                  : "Consensus";
                const actionLabel = alreadyAdded
                  ? `${sportsbook} ${market} ${side} for ${edge.away_team} at ${edge.home_team} is already on the Bet Slip`
                  : `Add ${sportsbook} ${market} ${side} for ${edge.away_team} at ${edge.home_team} to the Bet Slip`;

                return (
                  <tr
                    key={edgeOfferKey(edge)}
                    id={isAlternative && alternativeIndex === 0
                      ? alternativesId
                      : undefined}
                    style={{
                      borderTop: "1px solid var(--line-soft)",
                      background: isAlternative ? "var(--bg-2)" : undefined,
                    }}
                  >
                    <th
                      scope="row"
                      style={{
                        padding: "10px 12px 10px 0",
                        textAlign: "left",
                        fontWeight: 400,
                      }}
                    >
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        {isAlternative && <span className="dim">↳</span>}
                        <TeamMark abbr={edge.away_team} />
                        <span className="dim">@</span>
                        <TeamMark abbr={edge.home_team} />
                      </span>
                    </th>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      <div>{sportsbook}</div>
                      {!isAlternative && group.alternatives.length > 0 && (
                        <button
                          type="button"
                          aria-expanded={isExpanded}
                          aria-controls={alternativesId}
                          aria-label={`${isExpanded ? "Hide" : "View"} ${group.alternatives.length} other ${group.alternatives.length === 1 ? "offer" : "offers"} for ${market} ${side} in ${edge.away_team} at ${edge.home_team}`}
                          onClick={() => toggleGroup(group.id)}
                          style={offerToggleStyle}
                        >
                          {isExpanded
                            ? "Hide offers"
                            : `${group.alternatives.length} other ${group.alternatives.length === 1 ? "offer" : "offers"}`}
                        </button>
                      )}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      {edge.market_type}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>{edge.side}</td>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      {formatMarketContext(edge.market_type, edge.market_value)}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      {formatAmericanOdds(edge.american_odds)}
                    </td>
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
                        label={actionLabel}
                        onClick={() =>
                          add(
                            createGameBetLeg({
                              edge,
                              source: "betslip-edges",
                              addedAt: new Date().toISOString(),
                              referenceBankroll: data.bankroll ?? null,
                              referenceKellyMultiplier:
                                data.kelly_multiplier ?? null,
                            }),
                          )
                        }
                      />
                    </td>
                  </tr>
                );
              };

              return (
                <Fragment key={group.id}>
                  {renderOfferRow(group.best, false)}
                  {isExpanded &&
                    group.alternatives.map((edge, index) =>
                      renderOfferRow(edge, true, index),
                    )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    )}
    </div>
  );
}

const offerToggleStyle: React.CSSProperties = {
  marginTop: 4,
  padding: 0,
  border: "none",
  background: "transparent",
  color: "var(--accent)",
  cursor: "pointer",
  fontFamily: "var(--f-mono)",
  fontSize: 9,
};

function formatMarketContext(
  marketType: string,
  marketValue: number | null | undefined,
): string {
  if (marketValue == null) return "—";
  if (marketType === "moneyline") {
    return `${(marketValue * 100).toFixed(1)}% no-vig`;
  }
  if (marketType === "spread") {
    const sign = marketValue > 0 ? "+" : "";
    return `Home ${sign}${marketValue.toFixed(1)}`;
  }
  if (marketType === "total") {
    return `Total ${marketValue.toFixed(1)}`;
  }
  return marketValue.toFixed(1);
}

function formatAmericanOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`;
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
  label,
  onClick,
}: {
  disabled: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      aria-label={label}
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
