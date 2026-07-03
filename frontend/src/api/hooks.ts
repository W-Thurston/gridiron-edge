import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

/**
 * Fetches the current NFL week from /weeks/current.
 *
 * Returns react-query result: { data, isLoading, error, ... }.
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
