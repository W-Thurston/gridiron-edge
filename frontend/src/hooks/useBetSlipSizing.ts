import {
  useEffect,
  useState,
} from "react";
import {
  usePortfolioSummary,
} from "../api/hooks";
import {
  DEFAULT_BET_SLIP_SIZING,
  parseBetSlipSizingPreference,
  resolveBetSlipBankroll,
  updateBetSlipSizingPreference,
  type BetSlipBankrollMode,
  type BetSlipBankrollSource,
  type BetSlipSizingPreference,
} from "../utils/betSlipSizing";

const STORAGE_KEY =
  "hm-betslip-sizing-v1";

export type BetSlipSizingUpdate =
  Partial<
    Pick<
      BetSlipSizingPreference,
      | "bankrollMode"
      | "whatIfBankroll"
      | "kellyMultiplier"
    >
  >;

export type BetSlipSizingResult = {
  preference:
    BetSlipSizingPreference;
  bankroll: number | null;
  bankrollSource:
    BetSlipBankrollSource;
  trackedBankroll: number | null;
  bankrollMode:
    BetSlipBankrollMode;
  whatIfBankroll: number | null;
  kellyMultiplier: number;
  isTrackedBankrollLoading: boolean;
  trackedBankrollError:
    | Error
    | null;
  updateSizing: (
    update: BetSlipSizingUpdate,
  ) => void;
  setBankrollMode: (
    mode: BetSlipBankrollMode,
  ) => void;
  setWhatIfBankroll: (
    amount: number | null,
  ) => void;
  setKellyMultiplier: (
    multiplier: number,
  ) => void;
};

function loadInitialPreference():
  BetSlipSizingPreference {
  try {
    const stored =
      localStorage.getItem(
        STORAGE_KEY,
      );

    if (!stored) {
      return DEFAULT_BET_SLIP_SIZING;
    }

    return (
      parseBetSlipSizingPreference(
        JSON.parse(stored) as unknown,
      ) ?? DEFAULT_BET_SLIP_SIZING
    );
  } catch {
    return DEFAULT_BET_SLIP_SIZING;
  }
}

function normalizeError(
  error: unknown,
): Error | null {
  if (error == null) {
    return null;
  }

  if (error instanceof Error) {
    return error;
  }

  return new Error(String(error));
}

export function useBetSlipSizing():
  BetSlipSizingResult {
  const summary =
    usePortfolioSummary();

  const [preference, setPreference] =
    useState<BetSlipSizingPreference>(
      loadInitialPreference,
    );

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(preference),
      );
    } catch {
      // Ignore storage quota and
      // availability errors.
    }
  }, [preference]);

  const trackedBankroll =
    typeof summary.data?.bankroll ===
      "number" &&
    Number.isFinite(
      summary.data.bankroll,
    ) &&
    summary.data.bankroll >= 0
      ? summary.data.bankroll
      : null;

  const resolved =
    resolveBetSlipBankroll({
      preference,
      trackedBankroll,
    });

  const updateSizing = (
    update: BetSlipSizingUpdate,
  ) => {
    setPreference((current) =>
      updateBetSlipSizingPreference({
        current,
        update,
      }),
    );
  };

  const setBankrollMode = (
    mode: BetSlipBankrollMode,
  ) => {
    updateSizing({
      bankrollMode: mode,
    });
  };

  const setWhatIfBankroll = (
    amount: number | null,
  ) => {
    updateSizing({
      whatIfBankroll: amount,
    });
  };

  const setKellyMultiplier = (
    multiplier: number,
  ) => {
    updateSizing({
      kellyMultiplier: multiplier,
    });
  };

  return {
    preference,
    bankroll: resolved.amount,
    bankrollSource:
      resolved.source,
    trackedBankroll,
    bankrollMode:
      preference.bankrollMode,
    whatIfBankroll:
      preference.whatIfBankroll,
    kellyMultiplier:
      preference.kellyMultiplier,
    isTrackedBankrollLoading:
      summary.isLoading,
    trackedBankrollError:
      normalizeError(summary.error),
    updateSizing,
    setBankrollMode,
    setWhatIfBankroll,
    setKellyMultiplier,
  };
}
