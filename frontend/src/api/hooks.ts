import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

/**
 * Fetches the current NFL week from /weeks/current.
 */
export function useCurrentWeek() {
  return useQuery({
    queryKey: ["weeks-current"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/weeks/current");
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches games for a given season and week.
 * Both parameters are optional — API defaults to current.
 */
export function useGamesList(params: { season?: string; week?: number } = {}) {
  return useQuery({
    queryKey: ["games", params],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/games", {
        params: { query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches detail for a single game.
 */
export function useGame(gameId: string | null) {
  return useQuery({
    queryKey: ["game", gameId],
    queryFn: async () => {
      if (!gameId) throw new Error("gameId required");
      const { data, error } = await apiClient.GET("/games/{game_id}", {
        params: { path: { game_id: gameId } },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: gameId !== null,
  });
}
/**
 * Fetches the team rankings list.
 */
export function useTeamRankings(params: { season?: string } = {}) {
  return useQuery({
    queryKey: ["teams", params],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/teams", {
        params: { query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches detail for a single team.
 */
export function useTeamProfile(abbr: string | null, params: { season?: string } = {}) {
  return useQuery({
    queryKey: ["team", abbr, params],
    queryFn: async () => {
      if (!abbr) throw new Error("abbr required");
      const { data, error } = await apiClient.GET("/teams/{abbr}", {
        params: {
          path: { abbr },
          query: params,
        },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: abbr !== null,
  });
}

/**
 * Fetches Monte Carlo playoff projections.
 */
export function useProjections() {
  return useQuery({
    queryKey: ["projections"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/projections");
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}
/**
 * Fetches the props list.
 */
export function usePropsList(
  params: {
    season?: string;
    week?: number;
    stat_type?: string;
    position?: string;
    min_p_over?: number;
  } = {},
) {
  return useQuery({
    queryKey: ["props", params],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/props", {
        params: { query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches detail for a single prop.
 */
export function useProp(propId: string | null) {
  return useQuery({
    queryKey: ["prop", propId],
    queryFn: async () => {
      if (!propId) throw new Error("propId required");
      const { data, error } = await apiClient.GET("/props/{prop_id}", {
        params: { path: { prop_id: propId } },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: propId !== null,
  });
}

/**
 * Fetches the player-vs-defense comparison for a single prop.
 */
export function useComparePlayer(propId: string | null) {
  return useQuery({
    queryKey: ["compare-player", propId],
    queryFn: async () => {
      if (!propId) throw new Error("propId required");
      const { data, error } = await apiClient.GET(
        "/compare/player/{prop_id}",
        {
          params: { path: { prop_id: propId } },
        },
      );
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: propId !== null,
  });
}
/**
 * Fetches team-vs-team comparison.
 * Both team_a and team_b are required for the API to return data.
 * When either is missing, the hook is disabled.
 */
export function useCompareTeams(params: {
  team_a: string | null;
  team_b: string | null;
  season?: string;
}) {
  const enabled = params.team_a !== null && params.team_b !== null;
  return useQuery({
    queryKey: ["compare-teams", params],
    queryFn: async () => {
      if (!params.team_a || !params.team_b) {
        throw new Error("team_a and team_b required");
      }
      const { data, error } = await apiClient.GET("/compare/teams", {
        params: {
          query: {
            team_a: params.team_a,
            team_b: params.team_b,
            season: params.season,
          },
        },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled,
  });
}
/**
 * Fetches portfolio summary.
 */
export function usePortfolioSummary() {
  return useQuery({
    queryKey: ["portfolio-summary"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/portfolio/summary");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });
}

/**
 * Fetches open + recent bets from the ledger.
 */
export function usePortfolioBets() {
  return useQuery({
    queryKey: ["portfolio-bets"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/portfolio/bets");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });
}

/**
 * Fetches historical bankroll balance curve.
 */
export function usePortfolioCurve() {
  return useQuery({
    queryKey: ["portfolio-curve"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/portfolio/curve");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });
}

/**
 * Fetches raw bankroll transaction log.
 */
export function usePortfolioTransactions() {
  return useQuery({
    queryKey: ["portfolio-transactions"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/portfolio/transactions");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });
}

/**
 * Fetches performance splits by the given dimension.
 * Dimension defaults to "market_type" server-side.
 */
export function usePortfolioSplits(dimension?: "market_type" | "confidence_tier" | "model_type") {
  return useQuery({
    queryKey: ["portfolio-splits", dimension],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/portfolio/splits", {
        params: {
          query: dimension ? { dimension } : {},
        },
      });
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });
}
/**
 * Fetches ranked edges for a given week and min-EV threshold.
 */
export function useEdges(params: {
  season?: string;
  week?: number;
  min_ev?: number;
} = {}) {
  return useQuery({
    queryKey: ["edges", params],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/edges", {
        params: { query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches the skill-player roster for a season (Compare player picker).
 * Season optional — API defaults to latest.
 */
export function usePlayersList(params: { season?: number } = {}) {
  return useQuery({
    queryKey: ["players", params],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/players", {
        params: { query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
  });
}

/**
 * Fetches a player's per-game stat history for one season (bar chart).
 */
export function usePlayerHistory(
  playerId: string | null,
  params: { stat: string; season?: number; limit?: number },
) {
  return useQuery({
    queryKey: ["player-history", playerId, params],
    queryFn: async () => {
      if (!playerId) throw new Error("playerId required");
      const { data, error } = await apiClient.GET("/players/{player_id}/history", {
        params: { path: { player_id: playerId }, query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: playerId !== null && !!params.stat,
  });
}

/**
 * Fetches a team's allowed aggregates for a stat_type (all cohorts).
 */
export function useDefenseAllowed(
  team: string | null,
  params: { stat_type: string },
) {
  return useQuery({
    queryKey: ["defense-allowed", team, params],
    queryFn: async () => {
      if (!team) throw new Error("team required");
      const { data, error } = await apiClient.GET("/defense/{team}/allowed", {
        params: { path: { team }, query: params },
      });
      if (error) {
        throw new Error(JSON.stringify(error));
      }
      return data;
    },
    enabled: team !== null && !!params.stat_type,
  });
}
