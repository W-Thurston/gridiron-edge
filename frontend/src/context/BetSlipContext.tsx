import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";
import type { ReactNode } from "react";
import {
  parseBetLegsV2,
  parseBetLegV2,
  type BetLeg,
} from "../utils/betLegs";

export type { BetLeg } from "../utils/betLegs";

export type BetSlipMode = "single" | "parlay";

type BetSlipContextValue = {
  legs: BetLeg[];
  mode: BetSlipMode;
  add: (leg: BetLeg) => void;
  remove: (id: string) => void;
  clear: () => void;
  setMode: (mode: BetSlipMode) => void;
};

const BetSlipContext = createContext<
  BetSlipContextValue | undefined
>(undefined);

const LEGS_STORAGE_KEY = "hm-betslip-v2";
const MODE_STORAGE_KEY = "hm-betslip-mode-v2";

function loadInitialLegs(): BetLeg[] {
  try {
    const stored = localStorage.getItem(
      LEGS_STORAGE_KEY,
    );

    if (!stored) {
      return [];
    }

    return parseBetLegsV2(
      JSON.parse(stored) as unknown,
    );
  } catch {
    return [];
  }
}

function loadInitialMode(): BetSlipMode {
  try {
    const stored = localStorage.getItem(
      MODE_STORAGE_KEY,
    );

    if (
      stored === "single" ||
      stored === "parlay"
    ) {
      return stored;
    }
  } catch {
    // Ignore storage errors.
  }

  return "single";
}

export function BetSlipProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [legs, setLegs] =
    useState<BetLeg[]>(loadInitialLegs);
  const [mode, setModeInternal] =
    useState<BetSlipMode>(loadInitialMode);

  useEffect(() => {
    try {
      localStorage.setItem(
        LEGS_STORAGE_KEY,
        JSON.stringify(legs),
      );
    } catch {
      // Ignore quota errors.
    }
  }, [legs]);

  useEffect(() => {
    try {
      localStorage.setItem(
        MODE_STORAGE_KEY,
        mode,
      );
    } catch {
      // Ignore quota errors.
    }
  }, [mode]);

  const add = (leg: BetLeg) => {
    const parsed = parseBetLegV2(leg);

    if (!parsed) {
      return;
    }

    setLegs((previous) => {
      if (
        previous.some(
          (existing) =>
            existing.id === parsed.id,
        )
      ) {
        return previous;
      }

      return [...previous, parsed];
    });
  };

  const remove = (id: string) => {
    setLegs((previous) =>
      previous.filter(
        (leg) => leg.id !== id,
      ),
    );
  };

  const clear = () => setLegs([]);

  const setMode = (
    nextMode: BetSlipMode,
  ) => setModeInternal(nextMode);

  return (
    <BetSlipContext.Provider
      value={{
        legs,
        mode,
        add,
        remove,
        clear,
        setMode,
      }}
    >
      {children}
    </BetSlipContext.Provider>
  );
}

export function useBetSlip(): BetSlipContextValue {
  const context = useContext(BetSlipContext);

  if (!context) {
    throw new Error(
      "useBetSlip must be used inside a BetSlipProvider",
    );
  }

  return context;
}
