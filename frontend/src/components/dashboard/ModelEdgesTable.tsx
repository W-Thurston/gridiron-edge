import { useState } from "react";
import { useEdges } from "../../api/hooks";
import { useBetSlip } from "../../context/BetSlipContext";
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
 * 1. Fetch /edges (already sorted by EV)
 * 2. Filter by market_type based on active tab
 * 3. Render top 6 as table rows
 * 4. Each row: navigate to GameDetail on body click, add to slip via button
 *
 * Uses shared Pill primitive for filter tabs.
 */
export function ModelEdgesTable() {
  const [filter, setFilter] = useState<MarketFilter>("all");
  const { data, isLoading, error } = useEdges();
  const { navigate } = useNav();
  const { legs, add } = useBetSlip();

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

  const items = data?.items ?? [];
  const filtered = filter === "all"
    ? items
    : items.filter((e) => e.market_type === filter);
  const displayed = filtered.slice(0, 6);

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
              <th style={{ padding: "8px 12px 8px 0" }}>#</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Match</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Side</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Market</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Fair</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Cover Prob</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>EV</th>
              <th style={{ padding: "8px 0" }}></th>
            </tr>
          </thead>
          <tbody>
            {displayed.map((edge, i) => {
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
                market === "spread" ||
                market === "total"
                  ? edge.market_value ?? null
                  : null;

              const legId = buildGameBetLegId({
                gameId: edge.game_id,
                market,
                side,
                line,
              });
              const isPicked = legs.some((l) => l.id === legId);
              return (
                <tr
                  key={legId}
                  className="proj-row"
                  onClick={() => navigate("/games", { gameId: edge.game_id })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      navigate("/games", { gameId: edge.game_id });
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`View details for ${edge.away_team} at ${edge.home_team}`}
                  style={{
                    borderTop: "1px solid var(--line-soft)",
                    cursor: "pointer",
                  }}
                >
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                    {String(i + 1).padStart(2, "0")}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <TeamMark abbr={edge.away_team} size={18} />
                      <span className="dim">@</span>
                      <TeamMark abbr={edge.home_team} size={18} />
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>{edge.side}</td>
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-3)" }}>
                    {edge.market_type}
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
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isPicked) return;
                        add(
                          createGameBetLeg({
                            edge,
                            source:
                              "dashboard-model-edges",
                            addedAt:
                              new Date().toISOString(),
                            referenceBankroll:
                              data?.bankroll ?? null,
                            referenceKellyMultiplier:
                              data?.kelly_multiplier ??
                              null,
                          }),
                        );
                      }}
                      type="button"
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
            })}
          </tbody>
        </table>
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
