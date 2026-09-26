import { useState } from "react";

import { useComparables, useExplain, useGame } from "../api/hooks";
import type { components } from "../api/schema";
import { ErrorCard } from "../components/error/ErrorCard";
import { FieldValue } from "../components/field-status/FieldValue";
import type { FieldStatus } from "../components/field-status/types";
import { ComingSoonCard } from "../components/primitives/ComingSoonCard";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";
import { formatCalendarDate } from "../utils/datePresentation";

type ExplainFactor = components["schemas"]["ExplainFactor"];
type ComparableGame = components["schemas"]["ComparableGame"];
type GameComparables = components["schemas"]["GameComparables"];

const TOP_FACTORS_SHOWN = 8;

/**
 * "Why this number?" — the factor waterfall and comparable-games table,
 * both served from persisted evidence (ROADMAP.md Tier 3 #8/#10; U16).
 *
 * The credible band, outcome distribution, and market comparison remain
 * `ComingSoonCard` placeholders: they depend on the scenario engine
 * (Tier 7), which does not exist yet. Showing them as real content would
 * fabricate an interactivity the backend cannot back.
 */
export function ExplainPage() {
  const { route, navigate } = useNav();
  const gameId = route.params.gameId ?? null;

  const gameQuery = useGame(gameId);
  const explainQuery = useExplain(gameId);
  const comparablesQuery = useComparables(gameId);

  const backNav = (
    <div>
      <button
        type="button"
        onClick={() => navigate("/games")}
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
        ← Games
      </button>
    </div>
  );

  if (!gameId) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">No game selected.</div>
        </div>
      </div>
    );
  }

  if (gameQuery.isLoading || explainQuery.isLoading || comparablesQuery.isLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">Loading…</div>
        </div>
      </div>
    );
  }

  const error = gameQuery.error ?? explainQuery.error ?? comparablesQuery.error;
  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <ErrorCard
          error={error}
          onRetry={() => {
            gameQuery.refetch();
            explainQuery.refetch();
            comparablesQuery.refetch();
          }}
        />
      </div>
    );
  }

  const game = gameQuery.data;
  const explain = explainQuery.data;
  if (!game || !explain) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">No explanation data available.</div>
        </div>
      </div>
    );
  }

  const comparables = comparablesQuery.data;
  const explainStatus = explain._meta?.field_status ?? {};
  const comparablesStatus = comparables?._meta?.field_status ?? {};

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {backNav}

      <div
        className="hm-card"
        style={{
          padding: "20px 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 16,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <TeamMark abbr={game.away_team} size={32} />
          <span className="dim mono">@</span>
          <TeamMark abbr={game.home_team} size={32} />
          <div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>
              {game.away_team} @ {game.home_team}
            </div>
            <div className="dim mono" style={{ fontSize: 11 }}>
              {formatCalendarDate(game.game_date) ?? "Date unavailable"}
            </div>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="dim mono upper" style={{ fontSize: 10, letterSpacing: "0.08em" }}>
            Home win probability
          </div>
          <div className="mono tnum" style={{ fontSize: 28, fontWeight: 600 }}>
            <FieldValue
              value={
                explain.headline_win_prob != null
                  ? `${(explain.headline_win_prob * 100).toFixed(1)}%`
                  : null
              }
              status={explainStatus.headline_win_prob as FieldStatus | undefined}
            />
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1fr) 300px",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <FactorWaterfallCard
            factors={explain.factors}
            status={explainStatus.factors as FieldStatus | undefined}
          />
          <ComparablesCard
            comparables={comparables}
            status={comparablesStatus.comparables as FieldStatus | undefined}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <ComingSoonCard
            title="Credible band"
            status={explainStatus.band as FieldStatus | undefined}
          />
          <ComingSoonCard
            title="Outcome distribution"
            status={explainStatus.distribution as FieldStatus | undefined}
          />
          <ComingSoonCard
            title="Market comparison"
            status={explainStatus.market_implied as FieldStatus | undefined}
          />
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 style={{ margin: "0 0 12px", fontSize: 13, fontWeight: 600 }}>{children}</h3>;
}

function FactorWaterfallCard({
  factors,
  status,
}: {
  factors: ExplainFactor[] | null | undefined;
  status: FieldStatus | undefined;
}) {
  const [expanded, setExpanded] = useState(false);

  if (!factors || factors.length === 0) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <SectionTitle>What moved the number</SectionTitle>
        <FieldValue value={null} status={status} placeholder="No factors available" />
      </div>
    );
  }

  const baseline = factors.filter((factor) => factor.is_baseline);
  const rest = factors.filter((factor) => !factor.is_baseline);
  const sorted = [...rest].sort(
    (a, b) => Math.abs(b.log_odds_contribution ?? 0) - Math.abs(a.log_odds_contribution ?? 0),
  );
  const visible = expanded ? sorted : sorted.slice(0, TOP_FACTORS_SHOWN);
  const hiddenCount = sorted.length - visible.length;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <SectionTitle>What moved the number</SectionTitle>
      <p className="dim" style={{ fontSize: 12.5, margin: "0 0 14px", lineHeight: 1.5 }}>
        Each row is this feature's exact contribution to the model's log-odds
        output — a linear decomposition of the fitted estimator, not a causal
        effect. Sorted by size; the intercept is the model's neutral baseline.
      </p>
      {baseline.map((factor) => (
        <FactorRow key={factor.key ?? "intercept"} factor={factor} />
      ))}
      {visible.map((factor) => (
        <FactorRow key={factor.key} factor={factor} />
      ))}
      {hiddenCount > 0 && (
        <ShowMoreButton
          label={`Show ${hiddenCount} more factor${hiddenCount === 1 ? "" : "s"}`}
          onClick={() => setExpanded(true)}
        />
      )}
      {expanded && sorted.length > TOP_FACTORS_SHOWN && (
        <ShowMoreButton label="Show fewer factors" onClick={() => setExpanded(false)} />
      )}
    </div>
  );
}

function FactorRow({ factor }: { factor: ExplainFactor }) {
  const value = factor.log_odds_contribution ?? 0;
  const isPositive = value >= 0;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 12,
        padding: "7px 0",
        borderTop: "1px solid var(--line-soft)",
      }}
    >
      <span style={{ fontSize: 12.5, fontWeight: factor.is_baseline ? 600 : 400 }}>
        {factor.label ?? factor.key ?? "unknown factor"}
      </span>
      <span
        className="mono tnum"
        style={{
          fontSize: 12.5,
          fontWeight: 500,
          color: factor.is_baseline ? "var(--ink-3)" : isPositive ? "var(--pos)" : "var(--neg)",
        }}
      >
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function ComparablesCard({
  comparables,
  status,
}: {
  comparables: GameComparables | undefined;
  status: FieldStatus | undefined;
}) {
  const list = comparables?.comparables;

  if (!list) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <SectionTitle>Comparable historical games</SectionTitle>
        <FieldValue value={null} status={status} placeholder="No comparables available" />
      </div>
    );
  }

  if (list.length === 0) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <SectionTitle>Comparable historical games</SectionTitle>
        <p className="dim" style={{ fontSize: 12.5, margin: 0 }}>
          No historical game was similar enough to this matchup to qualify as
          a comparable — a genuinely unusual matchup, not missing evidence.
        </p>
      </div>
    );
  }

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <SectionTitle>Comparable historical games</SectionTitle>
      <p className="dim" style={{ fontSize: 12.5, margin: "0 0 12px" }}>
        {comparables?.sample_size ?? list.length} comparable game
        {(comparables?.sample_size ?? list.length) === 1 ? "" : "s"} · favorite won{" "}
        {formatPercent(comparables?.favorite_win_rate)} · covered{" "}
        {formatPercent(comparables?.favorite_cover_rate)}
      </p>
      <div>
        {list.map((game) => (
          <ComparableRow key={game.game_id} game={game} />
        ))}
      </div>
    </div>
  );
}

function ComparableRow({ game }: { game: ComparableGame }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "88px 1fr 70px 90px",
        gap: 10,
        alignItems: "center",
        padding: "8px 0",
        borderTop: "1px solid var(--line-soft)",
        fontSize: 12,
      }}
    >
      <span className="mono dim" style={{ fontSize: 10.5 }}>
        {game.season} · Wk {game.week}
      </span>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <TeamMark abbr={game.away_team} size={18} />
          <span className="dim mono" style={{ fontSize: 10 }}>
            @
          </span>
          <TeamMark abbr={game.home_team} size={18} />
          <span className="mono">
            {game.away_score}–{game.home_score}
          </span>
        </span>
        {game.favorite_team && (
          <span className="dim" style={{ fontSize: 10 }}>
            Favorite: {game.favorite_team}
            {game.spread_magnitude != null ? ` (−${game.spread_magnitude})` : ""}
          </span>
        )}
      </div>
      <span
        className="mono"
        style={{
          color:
            game.favorite_won == null
              ? "var(--ink-4)"
              : game.favorite_won
                ? "var(--pos)"
                : "var(--neg)",
        }}
      >
        {game.favorite_won == null ? "—" : game.favorite_won ? "Won" : "Lost"}
      </span>
      <span className="dim mono">
        {game.favorite_covered == null ? "—" : game.favorite_covered ? "Covered" : "No cover"}
      </span>
    </div>
  );
}

function ShowMoreButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mono dim"
      style={{
        marginTop: 10,
        width: "100%",
        padding: "6px",
        fontSize: 10.5,
        background: "transparent",
        border: "1px solid var(--line-soft)",
        borderRadius: 4,
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      {label}
    </button>
  );
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}
