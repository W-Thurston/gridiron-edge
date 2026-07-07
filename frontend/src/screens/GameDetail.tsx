import { useGame } from "../api/hooks";
import { ComingSoonCard } from "../components/games/ComingSoonCard";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { WinProbBand } from "../components/games/WinProbBand";
import { useNav } from "../context/NavContext";
import type { FieldStatus } from "../components/field-status/types";
import { ErrorCard } from "../components/error/ErrorCard";
import { TeamHero } from "../components/primitives/TeamHero";
import { useTeamByAbbr } from "../api/team_metadata_hook";
import { useEdges } from "../api/hooks";
import { useBetSlip } from "../context/BetSlipContext";
import { WhyLink } from "../components/primitives/WhyLink";
import { probToAmerican } from "../utils/odds";
import { TeamMark } from "../components/primitives/TeamMark";
import { useState } from "react";
import { Pill } from "../components/primitives/Pill";

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

      {/* Full-width header row: game header + model lean callout */}
      <div
        className="hm-card"
        style={{
          padding: "20px 24px 24px",
          borderBottom: "1px solid var(--line-soft)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 32,
        }}
      >
        <GameHeader
          awayTeam={data.away_team}
          homeTeam={data.home_team}
          gameDate={data.game_date}
          dayOfWeek={data.day_of_week}
        />
        <ModelLeanCallout
          gameId={data.game_id}
          awayTeam={data.away_team}
          homeTeam={data.home_team}
          confidenceTier={data.prediction?.confidence_tier ?? null}
        />
      </div>

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
          <LinesAndFairValueCard
            gameId={data.game_id}
            awayTeam={data.away_team}
            homeTeam={data.home_team}
            prediction={data.prediction}
          />
          <WinProbabilityCard
            prediction={data.prediction}
            awayTeam={data.away_team}
            homeTeam={data.home_team}
          />
          <TeamComparisonCard
            teamComparison={data.team_comparison}
            awayTeam={data.away_team}
            homeTeam={data.home_team}
          />
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
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto 1fr",
        alignItems: "center",
        gap: 32,
        flex: 1,
      }}
    >
      {/* Away team (right-oriented) */}
      <TeamHero
        team={{
          abbr: awayTeam,
          city: away?.city ?? undefined,
          name: awayName,
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
          name: homeName,
          primary_color: home?.primary_color ?? undefined,
          conference: home?.conference ?? undefined,
          division: home?.division ?? undefined,
        }}
        context="HOME"
        orientation="left"
        size={56}
      />
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
 * Win probability card — 2-column layout with:
 * - Left: home team prob band + label, away team prob band + label
 * - Right: projected score + margin
 *
 * Caveat callout (injury-related) is skipped — blocked on §5.3 injury
 * data source.
 */
function WinProbabilityCard({
  prediction,
  awayTeam,
  homeTeam,
}: {
  prediction:
    | {
        home_win_prob?: number | null;
        away_win_prob?: number | null;
        home_win_lo?: number | null;
        home_win_hi?: number | null;
        model_spread?: number | null;
        projected_home_score?: number | null;
        projected_away_score?: number | null;
      }
    | null
    | undefined;
  awayTeam: string;
  homeTeam: string;
}) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 16,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Win Probability
        </div>
        {prediction && (
          <div className="mono dim2" style={{ fontSize: 10 }}>
            with uncertainty band
          </div>
        )}
      </div>

      {prediction ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 24,
            alignItems: "center",
          }}
        >
          {/* Left column: Prob bands with team labels */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <TeamProbRow
              team={homeTeam}
              prob={prediction.home_win_prob}
              lo={prediction.home_win_lo}
              hi={prediction.home_win_hi}
            />
            <TeamProbRow
              team={awayTeam}
              prob={prediction.away_win_prob}
              lo={
                prediction.home_win_hi != null
                  ? 1 - prediction.home_win_hi
                  : undefined
              }
              hi={
                prediction.home_win_lo != null
                  ? 1 - prediction.home_win_lo
                  : undefined
              }
            />
          </div>

          {/* Right column: Projected score + margin */}
          <div style={{ borderLeft: "1px solid var(--line-soft)", paddingLeft: 24 }}>
            <div
              className="upper dim2"
              style={{
                fontSize: 10,
                letterSpacing: "0.08em",
                marginBottom: 8,
              }}
            >
              Projected Score
            </div>
            <div
              className="mono tnum"
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
              }}
            >
              {prediction.projected_away_score != null &&
              prediction.projected_home_score != null ? (
                <>
                  <span style={{ color: "var(--ink-2)" }}>
                    {awayTeam} {prediction.projected_away_score.toFixed(1)}
                  </span>
                  <span className="dim2" style={{ margin: "0 10px" }}>
                    —
                  </span>
                  <span style={{ color: "var(--pos)" }}>
                    {homeTeam} {prediction.projected_home_score.toFixed(1)}
                  </span>
                </>
              ) : (
                "—"
              )}
            </div>
            {prediction.model_spread != null && (
              <div
                className="mono dim"
                style={{ fontSize: 10.5, marginTop: 4 }}
              >
                Margin: {formatMargin(prediction.model_spread, awayTeam, homeTeam)}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="dim mono">No prediction available.</div>
      )}
    </div>
  );
}

/**
 * Single team row in the win probability card. Shows team abbrev + big %
 * + band range label + WinProbBand visualization.
 */
function TeamProbRow({
  team,
  prob,
  lo,
  hi,
}: {
  team: string;
  prob: number | null | undefined;
  lo: number | null | undefined;
  hi: number | null | undefined;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 6,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <TeamMark abbr={team} size={20} />
          <span style={{ fontWeight: 500 }}>{team}</span>
        </div>
        <span
          className="mono tnum"
          style={{
            fontWeight: 600,
            fontSize: 14,
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          {prob != null ? `${Math.round(prob * 100)}%` : "—"}
          {lo != null && hi != null && (
            <span
              className="dim2"
              style={{ fontWeight: 400, fontSize: 11 }}
            >
              · band {Math.round(lo * 100)}–{Math.round(hi * 100)}%
            </span>
          )}
        </span>
      </div>
      <WinProbBand
        homeWinProb={prob}
        homeWinLo={lo}
        homeWinHi={hi}
      />
    </div>
  );
}

/**
 * Format the margin display.
 *
 * model_spread is negative when home is favored, positive when away is favored.
 * Absolute value gives the magnitude.
 */
function formatMargin(
  modelSpread: number,
  awayTeam: string,
  homeTeam: string,
): string {
  const magnitude = Math.abs(modelSpread);
  const favoredTeam = modelSpread < 0 ? homeTeam : awayTeam;
  return `${favoredTeam} by ${magnitude.toFixed(1)}`;
}


/**
 * Model lean callout on the right side of the header. Shows top edge
 * for this game (recommendation, EV, confidence tier, WhyLink, slip button).
 *
 * Data flow:
 * 1. Fetch all edges via useEdges
 * 2. Filter to this game_id client-side
 * 3. Take top by EV
 * 4. Render recommendation + metadata
 *
 * Empty states:
 * - No edges available (odds blocked): shows "No model edge" muted
 * - Loading: shows nothing (avoids flash)
 * - Error: shows nothing
 */
function ModelLeanCallout({
  gameId,
  awayTeam,
  homeTeam,
  confidenceTier,
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  confidenceTier: string | null;
}) {
  const { data, isLoading, error } = useEdges();
  const { legs, add } = useBetSlip();

  if (isLoading || error) return null;

  const items = data?.items ?? [];
  const gameEdges = items
    .filter((e) => e.game_id === gameId)
    .sort((a, b) => (b.ev ?? 0) - (a.ev ?? 0));
  const topEdge = gameEdges[0];

  if (!topEdge) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: 4,
          minWidth: 200,
        }}
      >
        <div
          className="upper dim"
          style={{
            fontSize: 9.5,
            letterSpacing: "0.1em",
            color: "var(--ink-4)",
          }}
        >
          Model Lean
        </div>
        <div
          className="dim mono"
          style={{ fontSize: 12, marginTop: 4 }}
        >
          No model edge
        </div>
      </div>
    );
  }

  const legId = `game-detail-lean-${topEdge.game_id}`;
  const isPicked = legs.some((l) => l.id === legId);

  const handleAddSlip = () => {
    if (isPicked) return;
    add({
      id: legId,
      gameId: topEdge.game_id,
      market: topEdge.market_type as "moneyline" | "spread" | "total",
      side: topEdge.side as "home" | "away" | "over" | "under",
      odds: -110,
      awayTeam: awayTeam,
      homeTeam: homeTeam,
    });
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 4,
        minWidth: 200,
      }}
    >
      <span
        className="upper dim"
        style={{
          fontSize: 9.5,
          letterSpacing: "0.1em",
          color: "var(--ink-4)",
        }}
      >
        Model Lean
      </span>
      <span
        style={{
          fontSize: 18,
          fontWeight: 600,
          color: "var(--pos)",
        }}
      >
        {topEdge.side}
      </span>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 11,
          color: "var(--ink-3)",
        }}
      >
        <span className="mono">
          +{(topEdge.ev * 100).toFixed(1)}% EV
        </span>
        {confidenceTier && (
          <>
            <span>·</span>
            <ConfidenceTierPill tier={confidenceTier} />
          </>
        )}
        <WhyLink
          dot
          tone="pos"
          subject={{ kind: "rec", gameId: topEdge.game_id }}
        />
      </div>
      <button
        onClick={handleAddSlip}
        type="button"
        disabled={isPicked}
        style={{
          padding: "6px 14px",
          background: isPicked ? "var(--bg-3)" : "var(--pos)",
          color: isPicked ? "var(--ink-4)" : "var(--bg)",
          border: "none",
          borderRadius: 4,
          fontSize: 12,
          fontWeight: 600,
          fontFamily: "var(--f-sans)",
          cursor: isPicked ? "default" : "pointer",
          marginTop: 4,
        }}
      >
        {isPicked ? "✓ On slip" : "Add to bet slip"}
      </button>
    </div>
  );
}

/**
 * Lines & Model Fair Value card. 3-row table showing:
 * - Market row: blocked on W7 (odds ingest), all cells em-dash
 * - Gridiron Edge fair: model spread/total/moneyline from prediction data
 * - Recommendation: composed from /edges filtered to game_id
 *
 * Recommendation row has subtle green tint highlight.
 */
function LinesAndFairValueCard({
  gameId,
  awayTeam,
  homeTeam,
  prediction,
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  prediction: {
    home_win_prob?: number | null;
    away_win_prob?: number | null;
    model_spread?: number | null;
    model_total?: number | null;
  } | null | undefined;
}) {
  const { data: edgesData } = useEdges();

  // Filter edges to this game and split by market
  const gameEdges = (edgesData?.items ?? []).filter((e) => e.game_id === gameId);
  const spreadEdge = gameEdges
    .filter((e) => e.market_type === "spread")
    .sort((a, b) => (b.ev ?? 0) - (a.ev ?? 0))[0];
  const totalEdge = gameEdges
    .filter((e) => e.market_type === "total")
    .sort((a, b) => (b.ev ?? 0) - (a.ev ?? 0))[0];
  const mlEdge = gameEdges
    .filter((e) => e.market_type === "moneyline")
    .sort((a, b) => (b.ev ?? 0) - (a.ev ?? 0))[0];

  // Model fair values
  const modelSpread = prediction?.model_spread;
  const modelTotal = prediction?.model_total;
  const modelHomeML = prediction?.home_win_prob != null
    ? probToAmerican(prediction.home_win_prob)
    : null;
  const modelAwayML = prediction?.away_win_prob != null
    ? probToAmerican(prediction.away_win_prob)
    : null;

  return (
    <div className="hm-card" style={{ padding: 0 }}>
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid var(--line-soft)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Lines & Model Fair Value
        </div>
      </div>

      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
        }}
      >
        <thead>
          <tr style={{ borderBottom: "1px solid var(--line-soft)" }}>
            <th
              style={{
                padding: "10px 16px",
                textAlign: "left",
                fontSize: 10,
                color: "var(--ink-4)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                fontWeight: 400,
                width: 140,
              }}
            ></th>
            <th
              style={{
                padding: "10px 16px",
                textAlign: "left",
                fontSize: 10,
                color: "var(--ink-4)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                fontWeight: 400,
              }}
            >
              Spread
            </th>
            <th
              style={{
                padding: "10px 16px",
                textAlign: "left",
                fontSize: 10,
                color: "var(--ink-4)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                fontWeight: 400,
              }}
            >
              Total
            </th>
            <th
              style={{
                padding: "10px 16px",
                textAlign: "left",
                fontSize: 10,
                color: "var(--ink-4)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                fontWeight: 400,
              }}
            >
              Moneyline
            </th>
          </tr>
        </thead>
        <tbody>
          {/* Market row - blocked on W7 */}
          <tr style={{ borderBottom: "1px solid var(--line-soft)" }}>
            <td
              style={{
                padding: "12px 16px",
                color: "var(--ink-2)",
                fontWeight: 500,
              }}
            >
              Market
            </td>
            <td style={{ padding: "12px 16px", color: "var(--ink-2)" }}>—</td>
            <td style={{ padding: "12px 16px", color: "var(--ink-2)" }}>—</td>
            <td style={{ padding: "12px 16px", color: "var(--ink-2)" }}>—</td>
          </tr>

          {/* Model fair row */}
          <tr style={{ borderBottom: "1px solid var(--line-soft)" }}>
            <td
              style={{
                padding: "12px 16px",
                color: "var(--ink-2)",
                fontWeight: 500,
              }}
            >
              Gridiron Edge fair
            </td>
            <td style={{ padding: "12px 16px", fontFamily: "var(--f-mono)" }}>
              {formatSpreadDisplay(modelSpread, awayTeam, homeTeam)}
            </td>
            <td style={{ padding: "12px 16px", fontFamily: "var(--f-mono)" }}>
              {formatTotalDisplay(modelTotal)}
            </td>
            <td style={{ padding: "12px 16px", fontFamily: "var(--f-mono)" }}>
              {formatMLDisplay(modelAwayML, modelHomeML, awayTeam, homeTeam)}
            </td>
          </tr>

          {/* Recommendation row */}
          <tr
            style={{
              background: "color-mix(in oklab, var(--pos) 5%, transparent)",
            }}
          >
            <td
              style={{
                padding: "12px 16px",
                color: "var(--pos)",
                fontWeight: 500,
              }}
            >
              Recommendation
            </td>
            <td style={{ padding: "12px 16px" }}>
              <RecCell edge={spreadEdge} />
            </td>
            <td style={{ padding: "12px 16px" }}>
              <RecCell edge={totalEdge} />
            </td>
            <td style={{ padding: "12px 16px" }}>
              <RecCell edge={mlEdge} />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/**
 * Recommendation cell — either the edge or "No play" if no edge exists
 * for this market.
 */
function RecCell({
  edge,
}: {
  edge:
    | {
        side: string;
        ev?: number | null;
      }
    | undefined;
}) {
  if (!edge) {
    return (
      <span
        className="mono dim"
        style={{ fontSize: 11 }}
      >
        No play
      </span>
    );
  }

  return (
    <div>
      <div
        style={{
          color: "var(--pos)",
          fontWeight: 600,
          fontFamily: "var(--f-mono)",
        }}
      >
        {edge.side}
      </div>
      {edge.ev != null && (
        <div
          className="mono"
          style={{
            fontSize: 10.5,
            color: "var(--pos)",
            marginTop: 2,
          }}
        >
          +{(edge.ev * 100).toFixed(1)}% EV
        </div>
      )}
    </div>
  );
}

/**
 * Format model spread as two-line stack (away/home perspectives).
 *
 * Backend: negative = home favored (e.g., model_spread = -4.5)
 * Display:
 *   SF +4.5      (away gets +spread magnitude)
 *   BAL -4.5     (home gets -spread magnitude)
 */
function formatSpreadDisplay(
  spread: number | null | undefined,
  awayTeam: string,
  homeTeam: string,
): React.ReactNode {
  if (spread == null) return "—";
  const magnitude = Math.abs(spread);
  const awaySign = spread > 0 ? "-" : "+";
  const homeSign = spread > 0 ? "+" : "-";
  return (
    <>
      <div>
        {awayTeam} {awaySign}
        {magnitude.toFixed(1)}
      </div>
      <div>
        {homeTeam} {homeSign}
        {magnitude.toFixed(1)}
      </div>
    </>
  );
}

/**
 * Format model total as two-line stack (over/under).
 */
function formatTotalDisplay(total: number | null | undefined): React.ReactNode {
  if (total == null) return "—";
  return (
    <>
      <div>O {total.toFixed(1)}</div>
      <div>U {total.toFixed(1)}</div>
    </>
  );
}

/**
 * Format model moneyline as two-line stack (away/home).
 */
function formatMLDisplay(
  awayML: number | null,
  homeML: number | null,
  awayTeam: string,
  homeTeam: string,
): React.ReactNode {
  if (awayML == null || homeML == null) return "—";
  const formatML = (v: number) => (v > 0 ? `+${v}` : `${v}`);
  return (
    <>
      <div>
        {awayTeam} {formatML(awayML)}
      </div>
      <div>
        {homeTeam} {formatML(homeML)}
      </div>
    </>
  );
}

type CohortKey = "season" | "l4" | "home" | "away";

const COHORT_TABS: { key: CohortKey; label: string }[] = [
  { key: "season", label: "Season" },
  { key: "l4", label: "Last 4" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
];

const METRICS: {
  key: string;
  label: string;
  better: "higher" | "lower";
  fmt: (v: number) => string;
}[] = [
  { key: "off_epa_per_play", label: "Off. EPA/play", better: "higher", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_pass_epa", label: "Pass EPA/play", better: "higher", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_rush_epa", label: "Rush EPA/play", better: "higher", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "def_epa_per_play", label: "Def. EPA/play", better: "lower", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "def_rush_epa", label: "Def. Rush EPA", better: "lower", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) },
  { key: "off_third_down_pct", label: "3rd-down conv.", better: "higher", fmt: (v) => (v * 100).toFixed(1) + "%" },
  { key: "off_redzone_td_pct", label: "Red-zone TD %", better: "higher", fmt: (v) => (v * 100).toFixed(1) + "%" },
  { key: "turnover_diff", label: "Turnover diff", better: "higher", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(1) },
];

/**
 * Team Comparison card — 3-column layout with:
 * - Column 1: away team value + colored by better/worse
 * - Column 2: metric label (dim, centered)
 * - Column 3: home team value + colored by better/worse
 *
 * Consumes team_comparison field from /games/{id} (Step 7c data).
 * Nested dict: {team_abbr: {cohort: {metric: value}}}
 *
 * Cohort switcher via Pill primitives.
 * Empty state when no data or cohort not present.
 */
function TeamComparisonCard({
  teamComparison,
  awayTeam,
  homeTeam,
}: {
  teamComparison: { [key: string]: unknown } | null | undefined;
  awayTeam: string;
  homeTeam: string;
}) {
  const { navigate } = useNav();
  const [cohort, setCohort] = useState<CohortKey>("season");

  const awayTeamData = teamComparison?.[awayTeam] as
    | Record<string, Record<string, number>>
    | undefined;
  const homeTeamData = teamComparison?.[homeTeam] as
    | Record<string, Record<string, number>>
    | undefined;
  const awayData = awayTeamData?.[cohort];
  const homeData = homeTeamData?.[cohort];
  const hasData = awayData != null && homeData != null;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Team Comparison
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ display: "flex", gap: 6 }}>
            {COHORT_TABS.map((tab) => (
              <Pill
                key={tab.key}
                active={cohort === tab.key}
                onClick={() => setCohort(tab.key)}
              >
                {tab.label}
              </Pill>
            ))}
          </div>
          <button
            type="button"
            onClick={() =>
              navigate("/compare", { a: awayTeam, b: homeTeam })
            }
            className="mono"
            style={{
              background: "color-mix(in oklab, var(--info) 8%, transparent)",
              border: "1px solid color-mix(in oklab, var(--info) 30%, transparent)",
              color: "var(--info)",
              padding: "4px 10px",
              borderRadius: 4,
              fontSize: 10,
              cursor: "pointer",
              fontFamily: "var(--f-mono)",
              letterSpacing: "0.04em",
            }}
          >
            Open full comparison →
          </button>
        </div>
      </div>

      {!hasData ? (
        <div
          style={{
            padding: 24,
            textAlign: "center",
            color: "var(--ink-4)",
            fontSize: 12,
          }}
        >
          No comparison data for {COHORT_TABS.find((t) => t.key === cohort)?.label}.
        </div>
      ) : (
        <div style={{ display: "grid", gap: 8 }}>
          {METRICS.map((metric, i) => (
            <MetricRow
              key={metric.key}
              awayValue={awayData[metric.key]}
              homeValue={homeData[metric.key]}
              label={metric.label}
              better={metric.better}
              fmt={metric.fmt}
              first={i === 0}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Single row in team comparison. Colored by which team wins the metric.
 */
function MetricRow({
  awayValue,
  homeValue,
  label,
  better,
  fmt,
  first,
}: {
  awayValue: number | null | undefined;
  homeValue: number | null | undefined;
  label: string;
  better: "higher" | "lower";
  fmt: (v: number) => string;
  first: boolean;
}) {
  if (awayValue == null || homeValue == null) {
    return (
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          gap: 12,
          alignItems: "center",
          padding: "8px 0",
          borderTop: first ? "none" : "1px solid var(--line-soft)",
          fontSize: 11.5,
        }}
      >
        <span className="mono dim2" style={{ textAlign: "right" }}>—</span>
        <span
          className="dim"
          style={{
            textAlign: "center",
            fontSize: 10,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          {label}
        </span>
        <span className="mono dim2">—</span>
      </div>
    );
  }

  // Determine which team wins this metric
  const awayWins =
    better === "higher"
      ? awayValue > homeValue
      : awayValue < homeValue;
  const homeWins = awayValue !== homeValue && !awayWins;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto 1fr",
        gap: 12,
        alignItems: "center",
        padding: "8px 0",
        borderTop: first ? "none" : "1px solid var(--line-soft)",
        fontSize: 11.5,
      }}
    >
      <span
        className="mono"
        style={{
          textAlign: "right",
          color: awayWins ? "var(--pos)" : "var(--ink-2)",
          fontWeight: awayWins ? 600 : 400,
        }}
      >
        {fmt(awayValue)}
      </span>
      <span
        className="dim"
        style={{
          textAlign: "center",
          fontSize: 10,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
      <span
        className="mono"
        style={{
          color: homeWins ? "var(--pos)" : "var(--ink-2)",
          fontWeight: homeWins ? 600 : 400,
        }}
      >
        {fmt(homeValue)}
      </span>
    </div>
  );
}
