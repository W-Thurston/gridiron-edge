import { useGame } from "../api/hooks";
import { ComingSoonCard } from "../components/games/ComingSoonCard";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { TeamMark } from "../components/games/TeamMark";
import { WinProbBand } from "../components/games/WinProbBand";
import { useNav } from "../context/NavContext";
import type { FieldStatus } from "../components/field-status/types";
import { ErrorCard } from "../components/error/ErrorCard";

export function GameDetail() {
  const { route, navigate } = useNav();
  const gameId = route.params.gameId ?? null;
  const { data, isLoading, error, refetch } = useGame(gameId);

  if (!gameId) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">No game selected.</div>
      </div>
    );
  }

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

  if (isLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">Loading…</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <ErrorCard
          error={error}
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">No game data available.</div>
        </div>
      </div>
    );
  }

  const fieldStatus = data._meta?.field_status;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {backNav}

      {/* Header */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Game — {data.game_id}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 20,
          }}
        >
          <TeamMark abbr={data.away_team} />
          <span className="dim">@</span>
          <TeamMark abbr={data.home_team} />
        </div>
        <div
          className="mono dim"
          style={{ fontSize: 12, marginTop: 12, display: "flex", gap: 16 }}
        >
          <span>Season: {data.season ?? "—"}</span>
          <span>Week: {data.week ?? "—"}</span>
          <span>Date: {data.game_date ?? "—"}</span>
          {data.day_of_week && <span>{data.day_of_week}</span>}
        </div>
      </div>

      {/* Prediction */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Prediction
        </div>
        {data.prediction ? (
          <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
            <PredictionCell
              label="Home WP"
              value={
                data.prediction.home_win_prob != null
                  ? `${Math.round(data.prediction.home_win_prob * 100)}%`
                  : "—"
              }
            />
            <PredictionCell
              label="Away WP"
              value={
                data.prediction.away_win_prob != null
                  ? `${Math.round(data.prediction.away_win_prob * 100)}%`
                  : "—"
              }
            />
            <PredictionCell
              label="Uncertainty Band"
              value={
                <WinProbBand
                  homeWinProb={data.prediction.home_win_prob}
                  homeWinLo={data.prediction.home_win_lo}
                  homeWinHi={data.prediction.home_win_hi}
                />
              }
            />
            <PredictionCell
              label="Spread"
              value={
                data.prediction.model_spread != null
                  ? formatSpread(data.prediction.model_spread)
                  : "—"
              }
            />
            <PredictionCell
              label="Total"
              value={data.prediction.model_total?.toFixed(1) ?? "—"}
            />
            <PredictionCell
              label="Projected"
              value={
                data.prediction.projected_away_score != null &&
                data.prediction.projected_home_score != null
                  ? `${data.prediction.projected_away_score.toFixed(0)} — ${data.prediction.projected_home_score.toFixed(0)}`
                  : "—"
              }
            />
            <PredictionCell
              label="Confidence"
              value={
                <ConfidenceTierPill tier={data.prediction.confidence_tier} />
              }
            />
          </div>
        ) : (
          <div className="dim mono">No prediction available.</div>
        )}
      </div>

      {/* Scaffolded cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <ComingSoonCard
          title="Weather"
          status={fieldStatus?.weather as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Team Comparison"
          status={fieldStatus?.team_comparison as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Swing Factors"
          status={fieldStatus?.swing_factors as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Injuries"
          status={fieldStatus?.injuries as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Top Prop Edges"
          status={fieldStatus?.top_prop_edges as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

function PredictionCell({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div style={{ minWidth: 80 }}>
      <div className="upper dim2" style={{ fontSize: 10, marginBottom: 6 }}>
        {label}
      </div>
      <div className="mono tnum" style={{ fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}

function formatSpread(spread: number): string {
  const sign = spread > 0 ? "+" : "";
  return `${sign}${spread.toFixed(1)}`;
}
