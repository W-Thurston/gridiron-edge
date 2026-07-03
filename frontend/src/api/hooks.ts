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
