import { usePropsList } from "../../api/hooks";
import { useBetSlip } from "../../context/BetSlipContext";
import { useNav } from "../../context/NavContext";
import { TeamMark } from "../primitives/TeamMark";
import { WhyLink } from "../primitives/WhyLink";

/**
 * Compact 5-row list of top prop edges for the current week.
 *
 * Data flow:
 * 1. Fetch /props (unfiltered)
 * 2. Sort client-side by predicted_mean descending
 * 3. Take top 5
 * 4. Render each as a compact row: player name + stat + lean + line + model value
 *
 * Row click → PlayerProp detail. "+ Slip" adds prop as bet slip leg.
 * "See all →" navigates to Players Explorer.
 */
export function PropEdgesRail() {
  const { data, isLoading, error } = usePropsList({});
  const { navigate } = useNav();
  const { legs, add } = useBetSlip();

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Prop Edges
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Prop Edges
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load props.
        </div>
      </div>
    );
  }

  const items = data?.items ?? [];
  const sorted = [...items]
    .filter((p) => p.projection?.predicted_mean != null)
    .sort((a, b) => {
      const aMean = a.projection?.predicted_mean ?? 0;
      const bMean = b.projection?.predicted_mean ?? 0;
      return bMean - aMean;
    })
    .slice(0, 5);

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
          Prop Edges
        </div>
        <button
          type="button"
          onClick={() => navigate("/players")}
          className="mono dim"
          style={{
            background: "transparent",
            border: "none",
            padding: 0,
            cursor: "pointer",
            font: "inherit",
            fontSize: 11,
            color: "var(--ink-3)",
          }}
        >
          See all →
        </button>
      </div>

      {sorted.length === 0 && <EmptyState />}

      {sorted.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {sorted.map((prop, i) => (
            <PropRow
              key={prop.prop_id}
              prop={prop}
              legs={legs}
              add={add}
              navigate={navigate}
              isFirst={i === 0}
            />
          ))}
        </div>
      )}
    </div>
  );
}

type PropRowProps = {
  prop: {
    prop_id: string;
    player_name: string;
    position: string;
    team: string;
    stat_type: string;
    projection?: {
      predicted_mean?: number | null;
      predicted_std?: number | null;
    } | null;
    line_context?: {
      line?: number | null;
      lean?: string | null;
    } | null;
  };
  legs: Array<{ id: string }>;
  add: (leg: Parameters<ReturnType<typeof import("../../context/BetSlipContext").useBetSlip>["add"]>[0]) => void;
  navigate: (path: string, params?: Record<string, string>) => void;
  isFirst: boolean;
};

function PropRow({ prop, legs, add, navigate, isFirst }: PropRowProps) {
  const legId = `dash-prop-${prop.prop_id}`;
  const isPicked = legs.some((l) => l.id === legId);
  const statLabel = formatStatType(prop.stat_type);
  const lean = prop.line_context?.lean ?? null;
  const line = prop.line_context?.line ?? null;
  const modelMean = prop.projection?.predicted_mean ?? null;

  const handleClick = () => {
    navigate("/players", { propId: prop.prop_id });
  };

  const handleAdd = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isPicked) return;
    // Note: Prop bet leg shape needs to match BetLeg schema.
    // For now, encoding as prop with placeholder values.
    add({
      id: legId,
      gameId: prop.prop_id,
      market: "prop" as never, // Bet slip supports moneyline/spread/total; prop is aspirational
      side: (lean ?? "over") as "home" | "away" | "over" | "under",
      odds: -110,
      awayTeam: prop.team,
      homeTeam: prop.team,
    });
  };

  return (
    <div
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
      tabIndex={0}
      role="button"
      aria-label={`View details for ${prop.player_name} ${statLabel}`}
      style={{
        padding: "10px 0",
        borderTop: isFirst ? "none" : "1px solid var(--line-soft)",
        cursor: "pointer",
      }}
    >
      {/* Top row: player + pos + slip button */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 12,
          }}
        >
          <TeamMark abbr={prop.team} size={16} />
          <span style={{ color: "var(--ink)" }}>{prop.player_name}</span>
          <span className="mono dim2" style={{ fontSize: 10 }}>
            {prop.position}
          </span>
        </div>
        <button
          onClick={handleAdd}
          type="button"
          aria-label={isPicked ? "Prop on slip" : "Add prop to slip"}
          style={{
            padding: "2px 8px",
            background: isPicked ? "var(--bg-3)" : "var(--pos)",
            color: isPicked ? "var(--ink-4)" : "var(--bg)",
            border: "none",
            borderRadius: 3,
            fontSize: 10,
            fontWeight: 600,
            cursor: isPicked ? "default" : "pointer",
            fontFamily: "var(--f-sans)",
            flexShrink: 0,
          }}
        >
          {isPicked ? "✓" : "+"}
        </button>
      </div>

      {/* Bottom row: stat + lean + line + model */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: 11,
        }}
      >
        <span className="dim mono">{statLabel}</span>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 10,
          }}
        >
          <span
            style={{
              color:
                lean === "Over"
                  ? "var(--pos)"
                  : lean === "Under"
                    ? "var(--neg)"
                    : "var(--ink-3)",
              fontWeight: 500,
            }}
          >
            {lean ?? "—"}
          </span>
          <span className="mono tnum" style={{ color: "var(--ink-2)" }}>
            {line != null ? line.toFixed(1) : "—"}
          </span>
          <span className="mono dim">
            model{" "}
            <span style={{ color: "var(--ink)" }}>
              {modelMean != null ? modelMean.toFixed(1) : "—"}
            </span>
          </span>
          <WhyLink
            dot
            tone="pos"
            subject={{ kind: "prop", propId: prop.prop_id }}
          />
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div style={{ padding: 24, textAlign: "center" }}>
      <div className="dim mono" style={{ fontSize: 12, marginBottom: 8 }}>
        No prop projections yet.
      </div>
      <div className="mono dim2" style={{ fontSize: 11 }}>
        Run `gridiron props projections` to populate.
      </div>
    </div>
  );
}

function formatStatType(statType: string): string {
  // Convert "qb_pass_yards" → "Pass Yds"
  const map: Record<string, string> = {
    qb_pass_yards: "Pass Yds",
    qb_rush_yards: "Rush Yds",
    rb_rush_yards: "Rush Yds",
    wr_rec_yards: "Rec Yds",
    te_rec_yards: "Rec Yds",
  };
  return map[statType] ?? statType;
}
