import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { normalizeSelectedSportsbooks, type SportsbookMode } from "../utils/sportsbookPreferences";

export type OddsFormat = "american" | "decimal";

export type LineShoppingDisplay = {
  valueHighlights: boolean;
  positiveEv: boolean;
  preferredPositiveEv: boolean;
  bestLine: boolean;
  bestPrice: boolean;
  modelFavorite: boolean;
};

const DEFAULT_LINE_SHOPPING_DISPLAY: LineShoppingDisplay = {
  valueHighlights: true,
  positiveEv: true,
  preferredPositiveEv: true,
  bestLine: true,
  bestPrice: true,
  modelFavorite: true,
};

export type AppState = {
  oddsFormat: OddsFormat;
  bankroll: number;
  onboarded: boolean;
  alerts: number;
  sportsbookMode: SportsbookMode;
  selectedSportsbooks: string[];
  lineShoppingDisplay: LineShoppingDisplay;
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
  sportsbookMode: "all",
  selectedSportsbooks: [],
  lineShoppingDisplay: DEFAULT_LINE_SHOPPING_DISPLAY,
};

function loadInitialState(): AppState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<AppState>;
      const sportsbookMode = parsed.sportsbookMode === "selected" ? "selected" : "all";
      const selectedSportsbooks = normalizeSelectedSportsbooks(parsed.selectedSportsbooks);
      const lineShoppingDisplay = {
        ...DEFAULT_LINE_SHOPPING_DISPLAY,
        ...(parsed.lineShoppingDisplay ?? {}),
      };
      return {
        ...DEFAULT_STATE,
        ...parsed,
        sportsbookMode: sportsbookMode === "selected" && selectedSportsbooks.length > 0
          ? "selected"
          : "all",
        selectedSportsbooks,
        lineShoppingDisplay,
      };
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
