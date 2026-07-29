import { usePropsList } from "../../api/hooks";
import { useBetSlip } from "../../context/BetSlipContext";
import { useNav } from "../../context/NavContext";
import {
  buildPropBetLegId,
  createPropBetLeg,
  propSideFromLean,
} from "../../utils/betLegs";
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
  const { data, isLoading, error } =
    usePropsList({});
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
    game_id: string;
    player_id: string;
    player_name: string;
    position: string;
    team: string;
    stat_type: string;
    model_key: string;
    projection?: {
      predicted_mean?: number | null;
      predicted_std?: number | null;
    } | null;
    line_context?: {
      line?: number | null;
      p_over?: number | null;
      lean?: string | null;
      confidence_tier?: string | null;
    } | null;
  };
  legs: Array<{ id: string }>;
  add: (
    leg: Parameters<
      ReturnType<
        typeof import(
          "../../context/BetSlipContext"
        ).useBetSlip
      >["add"]
    >[0],
  ) => void;
  navigate: (
    path: string,
    params?: Record<string, string>,
  ) => void;
  isFirst: boolean;
};

function PropRow({
  prop,
  legs,
  add,
  navigate,
  isFirst,
}: PropRowProps) {
  const statLabel = formatStatType(
    prop.stat_type,
  );
  const lean =
    prop.line_context?.lean ?? null;
  const line =
    prop.line_context?.line ?? null;
  const modelMean =
    prop.projection?.predicted_mean ??
    null;
  const side = propSideFromLean(lean);

  const legId =
    side == null
      ? null
      : buildPropBetLegId({
          propId: prop.prop_id,
          side,
          line,
        });

  const isPicked =
    legId != null &&
    legs.some(
      (leg) => leg.id === legId,
    );

  const handleClick = () => {
    navigate("/players", { propId: prop.prop_id });
  };

  const handleAdd = (
    event: React.MouseEvent,
  ) => {
    event.stopPropagation();

    if (isPicked || side == null) {
      return;
    }

    add(
      createPropBetLeg({
        prop,
        side,
        source:
          "dashboard-prop-edges",
        addedAt:
          new Date().toISOString(),
      }),
    );
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
          type="button"
          onClick={handleAdd}
          disabled={
            side == null || isPicked
          }
          aria-label={
            side == null
              ? "No wager side available"
              : isPicked
                ? "Prop on slip"
                : "Add prop to slip"
          }
          style={{
            padding: "2px 8px",
            backgroundColor:
              side == null || isPicked
                ? "var(--bg-3)"
                : "var(--pos)",
            color:
              side == null || isPicked
                ? "var(--ink-4)"
                : "var(--bg)",
            border: "none",
            borderRadius: 3,
            fontSize: 10,
            fontWeight: 600,
            cursor:
              side == null || isPicked
                ? "not-allowed"
                : "pointer",
            fontFamily: "var(--f-sans)",
            flexShrink: 0,
          }}
        >
          {side == null
            ? "—"
            : isPicked
              ? "✓"
              : "+"}
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
