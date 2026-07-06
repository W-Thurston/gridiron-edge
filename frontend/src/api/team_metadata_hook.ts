import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

/**
 * Fetches team metadata for all 32 teams and caches it.
 * Used by TeamMark and TeamHero primitives to look up primary_color,
 * conference, division, etc.
 *
 * Data is stable per session — teams don't change often. Cache duration
 * matches other reference data (long stale time).
 */
export function useTeamMetadata() {
  return useQuery({
    queryKey: ["team-metadata"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/teams");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes; reference data
  });
}

/**
 * Look up a single team's metadata by abbreviation from cached data.
 * Returns null if not found or cache empty.
 */
export function useTeamByAbbr(abbr: string | null | undefined) {
  const { data } = useTeamMetadata();
  if (!abbr || !data?.items) return null;
  return data.items.find((t) => t.abbr === abbr) ?? null;
}
