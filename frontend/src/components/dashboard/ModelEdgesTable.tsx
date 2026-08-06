import { Fragment, useState } from "react";
import { useEdges } from "../../api/hooks";
import { useBetSlip } from "../../context/BetSlipContext";
import { useAppState } from "../../context/AppStateContext";
import {
  filterEdgesBySportsbook,
  groupEdgeOffers,
  sportsbookDisplayName,
} from "../../utils/sportsbookPreferences";
import { useNav } from "../../context/NavContext";
import { EdgeResultStatus } from "../field-status/EdgeResultStatus";
import { Pill } from "../primitives/Pill";
import { TeamMark } from "../primitives/TeamMark";
import { buildGameBetLegId, createGameBetLeg } from "../../utils/betLegs";

type MarketFilter = "all" | "moneyline" | "spread" | "total";

/**
 * Ranked table of model edges for the current week with filter tabs.
 *
 * Data flow:
 * 1. Fetch /edges and apply the persisted sportsbook preference
 * 2. Group eligible offers by game, market, and side
 * 3. Filter grouped winners by the active market tab
 * 4. Render the top 6 wager families with expandable alternative offers
 * 5. Preserve sportsbook-specific Bet Slip staging for every offer
 *
 * Uses shared Pill primitive for filter tabs.
 */
export function ModelEdgesTable() {
  const [filter, setFilter] = useState<MarketFilter>("all");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    () => new Set(),
  );
  const { data, isLoading, error } = useEdges();
  const { navigate } = useNav();
  const { legs, add } = useBetSlip();
  const { state } = useAppState();

  const tabs: { value: MarketFilter; label: string }[] = [
    { value: "all", label: "All" },
    { value: "spread", label: "Spread" },
    { value: "total", label: "Total" },
    { value: "moneyline", label: "Moneyline" },
  ];

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Model Edges
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Model Edges
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load edges.
        </div>
      </div>
    );
  }

  const items = filterEdgesBySportsbook(data?.items ?? [], state);
  const groups = groupEdgeOffers(items);
  const filtered = filter === "all"
    ? groups
    : groups.filter((group) => group.marketType === filter);
  const displayed = filtered.slice(0, 6);

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
        <div id="model-edges-heading" className="upper dim" style={{ fontSize: 10 }}>
          Model Edges
          {data?.week && <> · Wk {data.week}</>}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {tabs.map((tab) => (
            <Pill
              key={tab.value}
              active={filter === tab.value}
              onClick={() => setFilter(tab.value)}
            >
              {tab.label}
            </Pill>
          ))}
        </div>
      </div>

      {displayed.length === 0 && data && (
        items.length === 0 ? (
          <EdgeResultStatus diagnostics={data.diagnostics} />
        ) : (
          <MarketFilterEmptyState market={filter} />
        )
      )}

      {displayed.length > 0 && (
        <div
          className="market-table-scroll"
          role="region"
          aria-labelledby="model-edges-heading"
          tabIndex={0}
        >
        <table
          className="market-table mono tnum"
          style={{
            width: "100%",
            fontSize: 12,
            borderCollapse: "collapse",
          }}
        >
          <caption className="visually-hidden">
            Best model edge for each wager family with expandable sportsbook offers
          </caption>
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>#</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Match</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Side</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Market</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Sportsbook</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Odds</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Fair</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Cover Prob</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>EV</th>
              <th style={{ padding: "8px 0" }}></th>
            </tr>
          </thead>
          <tbody>
            {displayed.map((group, i) => {
              const isExpanded = expandedGroups.has(group.id);
              const renderOfferRow = (
                edge: typeof group.best,
                isAlternative: boolean,
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
                const isPicked = legs.some((leg) => leg.id === legId);
                const sportsbook = edge.sportsbook
                  ? sportsbookDisplayName(edge.sportsbook)
                  : "Consensus";
                const actionLabel = isPicked
                  ? `${sportsbook} ${market} ${side} for ${edge.away_team} at ${edge.home_team} is already on the Bet Slip`
                  : `Add ${sportsbook} ${market} ${side} for ${edge.away_team} at ${edge.home_team} to the Bet Slip`;

                return (
                  <tr
                    key={legId}
                    className="proj-row"
                    style={{
                      borderTop: "1px solid var(--line-soft)",
                      background: isAlternative ? "var(--bg-2)" : undefined,
                    }}
                  >
                    <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                      {isAlternative ? "↳" : String(i + 1).padStart(2, "0")}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      <button
                        type="button"
                        aria-label={`View details for ${edge.away_team} at ${edge.home_team}`}
                        onClick={() =>
                          navigate("/games", { gameId: edge.game_id })
                        }
                        style={matchupButtonStyle}
                      >
                        <TeamMark abbr={edge.away_team} size={18} />
                        <span className="dim">@</span>
                        <TeamMark abbr={edge.home_team} size={18} />
                      </button>
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>{edge.side}</td>
                    <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                      {edge.market_type}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                      <div>{sportsbook}</div>
                      {!isAlternative && group.alternatives.length > 0 && (
                        <button
                          type="button"
                          aria-expanded={isExpanded}
                          aria-label={`${isExpanded ? "Hide" : "View"} ${group.alternatives.length} other ${group.alternatives.length === 1 ? "offer" : "offers"} for ${edge.market_type} ${edge.side} in ${edge.away_team} at ${edge.home_team}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleGroup(group.id);
                          }}
                          onKeyDown={(event) => event.stopPropagation()}
                          style={offerToggleStyle}
                        >
                          {isExpanded ? "Hide offers" : `${group.alternatives.length} other ${group.alternatives.length === 1 ? "offer" : "offers"}`}
                        </button>
                      )}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                      {formatAmericanOdds(edge.american_odds)}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                      {formatFair(edge.model_value, edge.market_type)}
                    </td>
                    <td style={{ padding: "10px 12px 10px 0" }}>
                      {formatProbability(edge.cover_prob)}
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
                      +{(edge.ev * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "10px 0", textAlign: "right" }}>
                      <button
                        onClick={(event) => {
                          event.stopPropagation();
                          if (isPicked) return;
                          add(
                            createGameBetLeg({
                              edge,
                              source: "dashboard-model-edges",
                              addedAt: new Date().toISOString(),
                              referenceBankroll: data?.bankroll ?? null,
                              referenceKellyMultiplier:
                                data?.kelly_multiplier ?? null,
                            }),
                          );
                        }}
                        onKeyDown={(event) => event.stopPropagation()}
                        type="button"
                        disabled={isPicked}
                        aria-label={actionLabel}
                        style={{
                          padding: "3px 10px",
                          background: isPicked ? "var(--bg-3)" : "var(--pos)",
                          color: isPicked ? "var(--ink-4)" : "var(--bg)",
                          border: "none",
                          borderRadius: 3,
                          fontSize: 10,
                          fontWeight: 600,
                          cursor: isPicked ? "default" : "pointer",
                          fontFamily: "var(--f-sans)",
                        }}
                      >
                        {isPicked ? "✓" : "+"}
                      </button>
                    </td>
                  </tr>
                );
              };

              return (
                <Fragment key={group.id}>
                  {renderOfferRow(group.best, false)}
                  {isExpanded &&
                    group.alternatives.map((edge) =>
                      renderOfferRow(edge, true),
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

function MarketFilterEmptyState({ market }: { market: MarketFilter }) {
  return (
    <div style={{ padding: 24, textAlign: "center" }}>
      <div className="dim mono" style={{ fontSize: 12 }}>
        No positive {market} edges in this view.
      </div>
    </div>
  );
}

const matchupButtonStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: 0,
  border: "none",
  background: "transparent",
  color: "inherit",
  cursor: "pointer",
  font: "inherit",
};

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

function formatAmericanOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : String(odds);
}

function formatProbability(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatFair(
  value: number | null | undefined,
  marketType: string,
): string {
  if (value == null) return "—";
  if (marketType === "moneyline") {
    return formatProbability(value);
  }
  return value.toFixed(1);
}
