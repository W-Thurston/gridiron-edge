import { useState } from "react";

import { useEdges, useGame, usePropsList } from "../api/hooks";
import { useTeamByAbbr } from "../api/team_metadata_hook";
import { ErrorCard } from "../components/error/ErrorCard";
import type { components } from "../api/schema";
import type { FieldStatus } from "../components/field-status/types";
import { ComingSoonCard } from "../components/primitives/ComingSoonCard";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { Pill } from "../components/primitives/Pill";
import { TeamHero } from "../components/primitives/TeamHero";
import { TeamMark } from "../components/primitives/TeamMark";
import { WhyLink } from "../components/primitives/WhyLink";
import { useBetSlip } from "../context/BetSlipContext";
import { useNav } from "../context/NavContext";
import { probToAmerican } from "../utils/odds";
import {
  buildGameBetLegId,
  buildPropBetLegId,
  createGameBetLeg,
  createPropBetLeg,
  propSideFromLean,
} from "../utils/betLegs";

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
            win={data.win}
            spread={data.spread}
            total={data.total}
          />
          <WinProbabilityCard
            win={data.win}
            spread={data.spread}
            projectedScore={data.projected_score}
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
          <TopPropEdgesCard gameId={data.game_id} />
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
 * Win probability card — 2-column layout with:
 * - Left: home team prob band + label, away team prob band + label
 * - Right: projected score + margin
 *
 * Caveat callout (injury-related) is skipped — blocked on §5.3 injury
 * data source.
 */
function WinProbabilityCard({
  win,
  spread,
  projectedScore,
  awayTeam,
  homeTeam,
}: {
  win: components["schemas"]["WinPredictionBlock"];
  spread: components["schemas"]["SpreadPredictionBlock"];
  projectedScore: components["schemas"]["ProjectedScoreBlock"];
  awayTeam: string;
  homeTeam: string;
}) {
  const winAvailable =
    win.home_win_prob != null &&
    win.away_win_prob != null;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        className="upper dim"
        style={{ fontSize: 10, marginBottom: 16 }}
      >
        Win Probability
      </div>

      {winAvailable ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 24,
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <ProbabilityRow team={homeTeam} probability={win.home_win_prob} />
            <ProbabilityRow team={awayTeam} probability={win.away_win_prob} />
          </div>

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
              {projectedScore.away != null && projectedScore.home != null ? (
                <>
                  <span style={{ color: "var(--ink-2)" }}>
                    {awayTeam} {projectedScore.away.toFixed(1)}
                  </span>
                  <span className="dim2" style={{ margin: "0 10px" }}>
                    —
                  </span>
                  <span style={{ color: "var(--pos)" }}>
                    {homeTeam} {projectedScore.home.toFixed(1)}
                  </span>
                </>
              ) : (
                "—"
              )}
            </div>
            {spread.model_spread != null && (
              <div className="mono dim" style={{ fontSize: 10.5, marginTop: 4 }}>
                Margin: {formatMargin(spread.model_spread, awayTeam, homeTeam)}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="dim mono">No Win prediction available.</div>
      )}
    </div>
  );
}

function ProbabilityRow({
  team,
  probability,
}: {
  team: string;
  probability: number | null | undefined;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <TeamMark abbr={team} size={20} />
        <span style={{ fontWeight: 500 }}>{team}</span>
      </div>
      <span className="mono tnum" style={{ fontWeight: 600, fontSize: 14 }}>
        {probability != null ? `${Math.round(probability * 100)}%` : "—"}
      </span>
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
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
}) {
  const {
    data,
    isLoading,
    error,
  } = useEdges();
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

  const market =
    topEdge.market_type as
      | "moneyline"
      | "spread"
      | "total";

  const side =
    topEdge.side as
      | "home"
      | "away"
      | "over"
      | "under";

  const line =
    market === "spread" ||
    market === "total"
      ? topEdge.market_value ?? null
      : null;

  const legId = buildGameBetLegId({
    gameId: topEdge.game_id,
    market,
    side,
    line,
  });

  const isPicked = legs.some(
    (leg) => leg.id === legId,
  );

  const handleAddSlip = () => {
    if (isPicked) {
      return;
    }

    add(
      createGameBetLeg({
        edge: topEdge,
        source: "game-detail-lean",
        addedAt:
          new Date().toISOString(),
        referenceBankroll:
          data?.bankroll ?? null,
        referenceKellyMultiplier:
          data?.kelly_multiplier ??
          null,
      }),
    );
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
  win,
  spread,
  total,
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  win: components["schemas"]["WinPredictionBlock"];
  spread: components["schemas"]["SpreadPredictionBlock"];
  total: components["schemas"]["TotalPredictionBlock"];
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

  // Model fair values from independent persisted components
  const modelSpread = spread.model_spread;
  const modelTotal = total.model_total;
  const modelHomeML = win.home_win_prob != null
    ? probToAmerican(win.home_win_prob)
    : null;
  const modelAwayML = win.away_win_prob != null
    ? probToAmerican(win.away_win_prob)
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
/**
 * Top prop edges card — right rail. 4-row compact list of top prop
 * projections for this game, sorted by predicted_mean descending.
 *
 * Data flow:
 * 1. Fetch /props (unfiltered)
 * 2. Filter to this game_id client-side
 * 3. Sort by predicted_mean descending
 * 4. Take top 4
 *
 * Row click navigates to PlayerProp. Uses same pattern as Dashboard's
 * PropEdgesRail.
 */
function TopPropEdgesCard({ gameId }: { gameId: string }) {
  const { navigate } = useNav();
  const { legs, add } = useBetSlip();
  const { data, isLoading, error } = usePropsList({});

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Top Prop Edges
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Top Prop Edges
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load props.
        </div>
      </div>
    );
  }

  const items = data?.items ?? [];
  const gameProps = items
    .filter((p) => p.game_id === gameId)
    .filter((p) => p.projection?.predicted_mean != null)
    .sort((a, b) => {
      const aMean = a.projection?.predicted_mean ?? 0;
      const bMean = b.projection?.predicted_mean ?? 0;
      return bMean - aMean;
    })
    .slice(0, 4);

  const totalGameProps = items.filter((p) => p.game_id === gameId).length;

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
          Top Prop Edges
        </div>
        {totalGameProps > 0 && (
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
              fontSize: 10.5,
              color: "var(--ink-3)",
            }}
          >
            See all {totalGameProps} →
          </button>
        )}
      </div>

      {gameProps.length === 0 && (
        <div style={{ padding: 16, textAlign: "center" }}>
          <div className="dim mono" style={{ fontSize: 12, marginBottom: 6 }}>
            No prop projections yet.
          </div>
          <div className="mono dim2" style={{ fontSize: 10.5 }}>
            Run `gridiron props projections`.
          </div>
        </div>
      )}

      {gameProps.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {gameProps.map((prop, i) => (
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
  add: Parameters<
    ReturnType<
      typeof useBetSlip
    >["add"]
  > extends [infer Leg]
    ? (leg: Leg) => void
    : never;
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
  const confidenceTier =
    prop.line_context
      ?.confidence_tier ?? null;
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
          "game-detail-prop",
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
      {/* Top row: player + position + confidence tier */}
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
        {confidenceTier && (
          <ConfidenceTierPill tier={confidenceTier} />
        )}
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
            }}
          >
            {side == null
              ? "—"
              : isPicked
                ? "✓"
                : "+"}
          </button>
        </div>
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
