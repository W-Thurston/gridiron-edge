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
 * Resolve one team from cached metadata by canonical abbreviation or long name.
 *
 * The public function name is retained for existing callers, while schedule
 * and edge surfaces may pass the service-preserved team name.
 */
export function useTeamByAbbr(identity: string | null | undefined) {
  const { data } = useTeamMetadata();
  if (!identity || !data?.items) return null;
  return data.items.find(
    (team) => team.abbr === identity || team.name === identity,
  ) ?? null;
}
