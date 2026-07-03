import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { AppStateProvider } from "../context/AppStateContext";
import { BetSlipProvider } from "../context/BetSlipContext";
import { NavProvider } from "../context/NavContext";

/**
 * Wraps a component in all providers needed for tests.
 * Fresh QueryClient per wrapper so tests don't share cache.
 */
export function TestWrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <AppStateProvider>
        <BetSlipProvider>
          <NavProvider>{children}</NavProvider>
        </BetSlipProvider>
      </AppStateProvider>
    </QueryClientProvider>
  );
}
