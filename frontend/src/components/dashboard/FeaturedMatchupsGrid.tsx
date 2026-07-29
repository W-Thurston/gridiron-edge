import { useEdges, useGamesList } from "../../api/hooks";
import { useBetSlip } from "../../context/BetSlipContext";
import { useNav } from "../../context/NavContext";
import { ConfidenceTierPill } from "../games/ConfidenceTierPill";
import { TeamHero } from "../primitives/TeamHero";
import { WhyLink } from "../primitives/WhyLink";
import { WinProbBand } from "../games/WinProbBand";
import { useTeamByAbbr } from "../../api/team_metadata_hook";
import type { components } from "../../api/schema";
import { buildGameBetLegId, createGameBetLeg } from "../../utils/betLegs";


type EdgeApiRow =
  components["schemas"]["EdgeRow"];

/**
 * Grid of 3 featured game cards for the current week.
 *
 * Data flow:
 * 1. Fetch /edges (ranked by EV) and /games in parallel
 * 2. Join by game_id
 * 3. Take top 3 edges that have corresponding game prediction data
 * 4. Render each as a `FeaturedCard`
 *
 * Empty state: renders when no edges available (blocked on odds ingest).
 */
export function FeaturedMatchupsGrid() {
  const edgesResult = useEdges();
  const gamesResult = useGamesList();

  const isLoading = edgesResult.isLoading || gamesResult.isLoading;
  const error = edgesResult.error ?? gamesResult.error;

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Featured Matchups
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Featured Matchups
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load matchups.
        </div>
      </div>
    );
  }

  const edges = edgesResult.data?.items ?? [];
  const games = gamesResult.data?.items ?? [];

  // Join edges to games by game_id
  const featured = edges
    .slice(0, 20) // limit initial pool
    .map((edge) => {
      const game = games.find((g) => g.game_id === edge.game_id);
      if (!game || !game.prediction) return null;
      return { edge, game };
    })
    .filter((x): x is { edge: typeof edges[0]; game: typeof games[0] } => x !== null)
    .slice(0, 3);

  if (featured.length === 0) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Featured Matchups
        </div>
        <div style={{ padding: 24, textAlign: "center" }}>
          <div className="dim mono" style={{ fontSize: 12, marginBottom: 8 }}>
            No featured matchups yet.
          </div>
          <div
            className="mono dim2"
            style={{ fontSize: 11 }}
          >
            Run `gridiron ingest dk-odds` and `gridiron edges report` to populate.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Featured Matchups
        </div>
        <div className="mono dim2" style={{ fontSize: 11 }}>
          Top {featured.length} by EV
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${featured.length}, 1fr)`,
          gap: 12,
        }}
      >
        {featured.map(({ edge, game }) => (
          <FeaturedCard
            key={game.game_id}
            game={game}
            edge={edge}
            referenceBankroll={
              edgesResult.data?.bankroll ??
              null
            }
            referenceKellyMultiplier={
              edgesResult.data
                ?.kelly_multiplier ?? null
            }
          />
        ))}
      </div>
    </div>
  );
}

type FeaturedCardProps = {
  game: {
    game_id: string;
    game_date?: string | null;
    week?: number | null;
    away_team: string;
    home_team: string;
    prediction?: {
      home_win_prob?: number | null;
      away_win_prob?: number | null;
      home_win_lo?: number | null;
      home_win_hi?: number | null;
      model_spread?: number | null;
      confidence_tier?: string | null;
    } | null;
  };
  edge: EdgeApiRow;
  referenceBankroll: number | null;
  referenceKellyMultiplier: number | null;
};

function FeaturedCard({
  game,
  edge,
  referenceBankroll,
  referenceKellyMultiplier,
}: FeaturedCardProps) {
  const { navigate } = useNav();
  const { legs, add } = useBetSlip();
  const awayTeam = useTeamByAbbr(game.away_team);
  const homeTeam = useTeamByAbbr(game.home_team);
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

  const handleCardClick = () => {
    navigate("/games", { gameId: game.game_id });
  };

  const handleAddSlip = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isPicked) return;
    add(
      createGameBetLeg({
        edge,
        source:
          "dashboard-featured",
        addedAt:
          new Date().toISOString(),
        referenceBankroll,
        referenceKellyMultiplier,
      }),
    );
  };

  const gameDate = game.game_date ?? "TBD";
  const week = game.week != null ? `Wk ${game.week}` : "";

  return (
    <div
      onClick={handleCardClick}
      className="hm-card"
      style={{
        padding: 14,
        cursor: "pointer",
        transition: "border-color 90ms ease",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          fontSize: 10,
        }}
      >
        <span className="mono upper dim">{gameDate}</span>
        <span className="mono dim2">{week}</span>
      </div>

      {/* Team rows */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <TeamHero
          team={{
            abbr: game.away_team,
            city: awayTeam?.city ?? undefined,
            name: awayTeam?.name ?? undefined,
            primary_color: awayTeam?.primary_color ?? undefined,
          }}
          record="—"
          size={32}
        />
        <TeamHero
          team={{
            abbr: game.home_team,
            city: homeTeam?.city ?? undefined,
            name: homeTeam?.name ?? undefined,
            primary_color: homeTeam?.primary_color ?? undefined,
          }}
          record="—"
          size={32}
        />
      </div>

      {/* Win prob band */}
      <div>
        <div
          className="mono dim"
          style={{ fontSize: 10, marginBottom: 4 }}
        >
          Home win prob{" "}
          {game.prediction?.home_win_prob != null
            ? `${Math.round(game.prediction.home_win_prob * 100)}%`
            : "—"}
          {game.prediction?.home_win_lo != null &&
            game.prediction?.home_win_hi != null && (
              <>
                {" "}
                [
                {Math.round(game.prediction.home_win_lo * 100)}–
                {Math.round(game.prediction.home_win_hi * 100)}]
              </>
            )}
        </div>
        <WinProbBand
          homeWinProb={game.prediction?.home_win_prob}
          homeWinLo={game.prediction?.home_win_lo}
          homeWinHi={game.prediction?.home_win_hi}
        />
      </div>

      {/* Confidence + edge callout footer */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          paddingTop: 8,
          borderTop: "1px solid var(--line-soft)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            className="mono"
            style={{ fontSize: 11, color: "var(--pos)" }}
          >
            ↗ {edge.side}
          </span>
          <span
            className="mono"
            style={{ fontSize: 11, color: "var(--pos)" }}
          >
            +{(edge.ev * 100).toFixed(1)}% EV
          </span>
          <ConfidenceTierPill tier={game.prediction?.confidence_tier ?? null} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <WhyLink
            dot
            tone="pos"
            subject={{ kind: "rec", gameId: game.game_id }}
          />
          <button
            onClick={handleAddSlip}
            type="button"
            style={{
              padding: "4px 10px",
              background: isPicked ? "var(--bg-3)" : "var(--pos)",
              color: isPicked ? "var(--ink-4)" : "var(--bg)",
              border: "none",
              borderRadius: 3,
              fontSize: 11,
              fontWeight: 600,
              cursor: isPicked ? "default" : "pointer",
              fontFamily: "var(--f-sans)",
            }}
          >
            {isPicked ? "✓ On slip" : "+ Slip"}
          </button>
        </div>
      </div>
    </div>
  );
}
