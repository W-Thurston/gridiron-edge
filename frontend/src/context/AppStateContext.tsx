import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type OddsFormat = "american" | "decimal";

export type AppState = {
  oddsFormat: OddsFormat;
  bankroll: number;
  onboarded: boolean;
  alerts: number;
};

type AppStateContextValue = {
  state: AppState;
  setState: (partial: Partial<AppState>) => void;
};

const AppStateContext = createContext<AppStateContextValue | undefined>(
  undefined,
);

const STORAGE_KEY = "hm-app";

const DEFAULT_STATE: AppState = {
  oddsFormat: "american",
  bankroll: 12480.55,
  onboarded: true,
  alerts: 4,
};

function loadInitialState(): AppState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<AppState>;
      // Merge with defaults so newly added keys don't break existing users.
      return { ...DEFAULT_STATE, ...parsed };
    }
  } catch {
    // Ignore parse errors.
  }
  return DEFAULT_STATE;
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, setStateInternal] = useState<AppState>(loadInitialState);

  // Persist to localStorage on every change.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Ignore quota errors.
    }
  }, [state]);

  const setState = (partial: Partial<AppState>) => {
    setStateInternal((prev) => ({ ...prev, ...partial }));
  };

  return (
    <AppStateContext.Provider value={{ state, setState }}>
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState(): AppStateContextValue {
  const ctx = useContext(AppStateContext);
  if (!ctx) {
    throw new Error("useAppState must be used inside an AppStateProvider");
  }
  return ctx;
}
