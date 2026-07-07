import { useGame } from "../api/hooks";
import { ComingSoonCard } from "../components/games/ComingSoonCard";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { WinProbBand } from "../components/games/WinProbBand";
import { useNav } from "../context/NavContext";
import type { FieldStatus } from "../components/field-status/types";
import { ErrorCard } from "../components/error/ErrorCard";
import { TeamHero } from "../components/primitives/TeamHero";
import { useTeamByAbbr } from "../api/team_metadata_hook";

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
      <GameHeader
        awayTeam={data.away_team}
        homeTeam={data.home_team}
        gameDate={data.game_date}
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
 * Full-width game header. Renders TeamHero components for both teams
 * framing a center block with kick datetime, venue, and weather.
 *
 * Model lean callout (recommendation + EV + WhyLink + slip button)
 * ships in Substep 2b.
 */
function GameHeader({
  awayTeam,
  homeTeam,
  gameDate,
  dayOfWeek,
}: {
  awayTeam: string;
  homeTeam: string;
  gameDate: string | null | undefined;
  dayOfWeek: string | null | undefined;
}) {
  const away = useTeamByAbbr(awayTeam);
  const home = useTeamByAbbr(homeTeam);
  const awayName = stripCityPrefix(away?.name ?? undefined, away?.city ?? undefined);
  const homeName = stripCityPrefix(home?.name ?? undefined, home?.city ?? undefined);

  return (
    <div
      className="hm-card"
      style={{
        padding: "20px 24px 24px",
        borderBottom: "1px solid var(--line-soft)",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
          gap: 32,
        }}
      >
        {/* Away team (right-oriented) */}
        <TeamHero
          team={{
            abbr: awayTeam,
            city: away?.city ?? undefined,
            name: awayName,                              // ← updated
            primary_color: away?.primary_color ?? undefined,
            conference: away?.conference ?? undefined,
            division: away?.division ?? undefined,
          }}
          context="AWAY"
          orientation="right"
          size={56}
        />

        {/* Center block */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: 180,
            gap: 4,
          }}
        >
          <div
            className="mono upper"
            style={{
              fontSize: 10,
              color: "var(--ink-3)",
              letterSpacing: "0.1em",
            }}
          >
            {formatKickLabel(gameDate, dayOfWeek)}
          </div>
          <div
            style={{
              fontFamily: "var(--f-serif)",
              fontSize: 24,
              fontStyle: "italic",
              color: "var(--ink)",
              lineHeight: 1,
            }}
          >
            at
          </div>
          <div
            className="mono"
            style={{
              fontSize: 10.5,
              color: "var(--ink-3)",
              marginTop: 4,
            }}
          >
            — · —
          </div>
          <div
            className="mono"
            style={{
              fontSize: 10.5,
              color: "var(--ink-3)",
            }}
          >
            —
          </div>
        </div>

        {/* Home team (left-oriented) */}
        <TeamHero
          team={{
            abbr: homeTeam,
            city: home?.city ?? undefined,
            name: homeName,                              // ← updated
            primary_color: home?.primary_color ?? undefined,
            conference: home?.conference ?? undefined,
            division: home?.division ?? undefined,
          }}
          context="HOME"
          orientation="left"
          size={56}
        />
      </div>
    </div>
  );
}

/**
 * Format game date + day-of-week into prototype-style compact kick label.
 *
 * Examples:
 * - gameDate "2026-02-08", dayOfWeek "Sunday" → "SUN · FEB 8"
 * - gameDate "2026-02-08", dayOfWeek null → "FEB 8"
 * - gameDate null → "—"
 *
 * When we get real kick times from the schedule, this expands to include
 * time-of-day (e.g., "SUN · FEB 8 · 4:25 PM ET").
 */
function formatKickLabel(
  gameDate: string | null | undefined,
  dayOfWeek: string | null | undefined,
): string {
  if (!gameDate) return "—";

  const dateObj = new Date(gameDate);
  if (isNaN(dateObj.getTime())) return "—";

  const monthNames = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
  ];
  const month = monthNames[dateObj.getUTCMonth()];
  const day = dateObj.getUTCDate();

  const dayShort = dayOfWeek
    ? dayOfWeek.slice(0, 3).toUpperCase()
    : null;

  const parts = [];
  if (dayShort) parts.push(dayShort);
  parts.push(`${month} ${day}`);
  return parts.join(" · ");
}

/**
 * Strip city prefix from team name when both are provided separately.
 *
 * Backend returns `name` as the full long form ("Seattle Seahawks") but
 * the metadata `city` is also exposed separately ("Seattle"). Without
 * this, TeamHero would render "Seattle Seattle Seahawks".
 *
 * Examples:
 * - name "Seattle Seahawks", city "Seattle" → "Seahawks"
 * - name "New England Patriots", city "New England" → "Patriots"
 * - name "Seahawks", city "Seattle" → "Seahawks" (already split)
 * - name undefined → undefined
 */
function stripCityPrefix(
  name: string | undefined,
  city: string | undefined,
): string | undefined {
  if (!name) return undefined;
  if (!city) return name;
  const prefix = `${city} `;
  if (name.startsWith(prefix)) {
    return name.slice(prefix.length);
  }
  return name;
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
