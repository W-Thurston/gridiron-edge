import { useGame } from "../api/hooks";
import { ComingSoonCard } from "../components/games/ComingSoonCard";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { TeamMark } from "../components/primitives/TeamMark";
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

      {/* Full-width header — Tier 2 will replace with composed team hero header */}
      <GameHeaderPlaceholder
        awayTeam={data.away_team}
        homeTeam={data.home_team}
        gameId={data.game_id}
        gameDate={data.game_date}
        season={data.season}
        week={data.week}
        dayOfWeek={data.day_of_week}
      />

      {/* 2-column grid: main content + right rail */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "3fr 2fr",
          gap: 16,
        }}
      >
        {/* Main column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <SectionPlaceholder title="Lines & Model Fair Value" />
          <WinProbabilityCard prediction={data.prediction} />
          <SectionPlaceholder title="Team Comparison" />
        </div>

        {/* Right rail */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <SectionPlaceholder title="Top Prop Edges" />
          <ComingSoonCard
            title="Swing Factors"
            status={fieldStatus?.swing_factors as FieldStatus | undefined}
          />
          <ComingSoonCard
            title="Injuries"
            status={fieldStatus?.injuries as FieldStatus | undefined}
          />
        </div>
      </div>
    </div>
  );
}

/**
 * Full-width game header placeholder. Tier 2 will replace with
 * TeamHero-based composed header.
 */
function GameHeaderPlaceholder({
  awayTeam,
  homeTeam,
  gameId,
  gameDate,
  season,
  week,
  dayOfWeek,
}: {
  awayTeam: string;
  homeTeam: string;
  gameId: string;
  gameDate: string | null | undefined;
  season: string | null | undefined;
  week: number | null | undefined;
  dayOfWeek: string | null | undefined;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Game — {gameId}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          fontSize: 20,
        }}
      >
        <TeamMark abbr={awayTeam} />
        <span className="dim">@</span>
        <TeamMark abbr={homeTeam} />
      </div>
      <div
        className="mono dim"
        style={{ fontSize: 12, marginTop: 12, display: "flex", gap: 16 }}
      >
        <span>Season: {season ?? "—"}</span>
        <span>Week: {week ?? "—"}</span>
        <span>Date: {gameDate ?? "—"}</span>
        {dayOfWeek && <span>{dayOfWeek}</span>}
      </div>
    </div>
  );
}

/**
 * Titled empty card for sections in-progress this workstream.
 * Different from ComingSoonCard which is used for cross-workstream
 * blocked sections.
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
        Coming in Tier 2/3
      </div>
    </div>
  );
}

/**
 * Prediction cells rendered as compact win probability card.
 * Tier 3b replaces with real 2-band + projected score composition.
 */
function WinProbabilityCard({
  prediction,
}: {
  prediction:
    | {
        home_win_prob?: number | null;
        away_win_prob?: number | null;
        home_win_lo?: number | null;
        home_win_hi?: number | null;
        confidence_tier?: string | null;
        model_spread?: number | null;
        model_total?: number | null;
        projected_home_score?: number | null;
        projected_away_score?: number | null;
      }
    | null
    | undefined;
}) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Win Probability
      </div>
      {prediction ? (
        <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
          <PredictionCell
            label="Home WP"
            value={
              prediction.home_win_prob != null
                ? `${Math.round(prediction.home_win_prob * 100)}%`
                : "—"
            }
          />
          <PredictionCell
            label="Away WP"
            value={
              prediction.away_win_prob != null
                ? `${Math.round(prediction.away_win_prob * 100)}%`
                : "—"
            }
          />
          <PredictionCell
            label="Uncertainty Band"
            value={
              <WinProbBand
                homeWinProb={prediction.home_win_prob}
                homeWinLo={prediction.home_win_lo}
                homeWinHi={prediction.home_win_hi}
              />
            }
          />
          <PredictionCell
            label="Spread"
            value={
              prediction.model_spread != null
                ? formatSpread(prediction.model_spread)
                : "—"
            }
          />
          <PredictionCell
            label="Total"
            value={prediction.model_total?.toFixed(1) ?? "—"}
          />
          <PredictionCell
            label="Projected"
            value={
              prediction.projected_away_score != null &&
              prediction.projected_home_score != null
                ? `${prediction.projected_away_score.toFixed(0)} — ${prediction.projected_home_score.toFixed(0)}`
                : "—"
            }
          />
          <PredictionCell
            label="Confidence"
            value={<ConfidenceTierPill tier={prediction.confidence_tier} />}
          />
        </div>
      ) : (
        <div className="dim mono">No prediction available.</div>
      )}
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
