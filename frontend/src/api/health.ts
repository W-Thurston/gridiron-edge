import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

/**
 * Checks if the API is reachable. Uses /weeks/current as a canary
 * because it's cheap and always available.
 *
 * Retries once with exponential backoff. Caches result for 30 seconds
 * so the banner doesn't flicker.
 */
export function useApiHealth() {
  return useQuery({
    queryKey: ["api-health"],
    queryFn: async () => {
      const { error } = await apiClient.GET("/weeks/current");
      if (error) {
        throw new Error("API not reachable");
      }
      return { healthy: true };
    },
    staleTime: 30_000,
    retry: 1,
    // Don't refetch aggressively on health check.
    refetchInterval: 30_000,
  });
}
