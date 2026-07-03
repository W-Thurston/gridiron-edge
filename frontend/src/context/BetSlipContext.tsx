import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

/** A single bet leg in the slip. */
export type BetLeg = {
  /** Unique identifier for dedup. Composed from game_id + market + side. */
  id: string;
  gameId: string;
  market: "moneyline" | "spread" | "total";
  side: "home" | "away" | "over" | "under";
  /** American odds. */
  odds: number;
  /** Optional line value (for spread/total). */
  line?: number;
  /** Display metadata — not used for logic, just for rendering. */
  awayTeam: string;
  homeTeam: string;
};

export type BetSlipMode = "single" | "parlay";

type BetSlipContextValue = {
  legs: BetLeg[];
  mode: BetSlipMode;
  add: (leg: BetLeg) => void;
  remove: (id: string) => void;
  clear: () => void;
  setMode: (mode: BetSlipMode) => void;
};

const BetSlipContext = createContext<BetSlipContextValue | undefined>(
  undefined,
);

const LEGS_STORAGE_KEY = "hm-betslip";
const MODE_STORAGE_KEY = "hm-betslip-mode";

function loadInitialLegs(): BetLeg[] {
  try {
    const stored = localStorage.getItem(LEGS_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) return parsed as BetLeg[];
    }
  } catch {
    // Ignore parse errors.
  }
  return [];
}

function loadInitialMode(): BetSlipMode {
  try {
    const stored = localStorage.getItem(MODE_STORAGE_KEY);
    if (stored === "single" || stored === "parlay") return stored;
  } catch {
    // Ignore.
  }
  return "single";
}

export function BetSlipProvider({ children }: { children: ReactNode }) {
  const [legs, setLegs] = useState<BetLeg[]>(loadInitialLegs);
  const [mode, setModeInternal] = useState<BetSlipMode>(loadInitialMode);

  // Persist legs.
  useEffect(() => {
    try {
      localStorage.setItem(LEGS_STORAGE_KEY, JSON.stringify(legs));
    } catch {
      // Ignore quota errors.
    }
  }, [legs]);

  // Persist mode.
  useEffect(() => {
    try {
      localStorage.setItem(MODE_STORAGE_KEY, mode);
    } catch {
      // Ignore.
    }
  }, [mode]);

  const add = (leg: BetLeg) => {
    setLegs((prev) => {
      // Dedupe by id — adding an existing leg is a no-op.
      if (prev.some((l) => l.id === leg.id)) return prev;
      return [...prev, leg];
    });
  };

  const remove = (id: string) => {
    setLegs((prev) => prev.filter((l) => l.id !== id));
  };

  const clear = () => setLegs([]);

  const setMode = (nextMode: BetSlipMode) => setModeInternal(nextMode);

  return (
    <BetSlipContext.Provider value={{ legs, mode, add, remove, clear, setMode }}>
      {children}
    </BetSlipContext.Provider>
  );
}

export function useBetSlip(): BetSlipContextValue {
  const ctx = useContext(BetSlipContext);
  if (!ctx) {
    throw new Error("useBetSlip must be used inside a BetSlipProvider");
  }
  return ctx;
}
