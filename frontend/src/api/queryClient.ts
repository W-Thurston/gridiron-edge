import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Data is considered fresh for 60 seconds. Cached responses
      // returned instantly during that window; refetch happens after.
      staleTime: 60_000,
      // Don't refetch on window focus — this app doesn't have
      // real-time data flowing in.
      refetchOnWindowFocus: false,
      // Retry once on failure. Second retry adds latency without
      // meaningful reliability gain for local dev.
      retry: 1,
    },
  },
});
