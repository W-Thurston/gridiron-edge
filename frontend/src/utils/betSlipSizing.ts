export const BET_SLIP_SIZING_VERSION = 1 as const;

export const DEFAULT_KELLY_MULTIPLIER = 0.25;

export type BetSlipBankrollMode =
  | "tracked"
  | "what-if";

export type BetSlipSizingPreference = {
  version: typeof BET_SLIP_SIZING_VERSION;
  bankrollMode: BetSlipBankrollMode;
  whatIfBankroll: number | null;
  kellyMultiplier: number;
};

export type BetSlipBankrollSource =
  | "tracked"
  | "what-if"
  | "unavailable";

export type ResolvedBetSlipBankroll = {
  amount: number | null;
  source: BetSlipBankrollSource;
};

export const DEFAULT_BET_SLIP_SIZING: BetSlipSizingPreference =
  {
    version: BET_SLIP_SIZING_VERSION,
    bankrollMode: "tracked",
    whatIfBankroll: null,
    kellyMultiplier:
      DEFAULT_KELLY_MULTIPLIER,
  };

export function parseBetSlipSizingPreference(
  value: unknown,
): BetSlipSizingPreference | null {
  if (!isRecord(value)) {
    return null;
  }

  if (
    value.version !==
    BET_SLIP_SIZING_VERSION
  ) {
    return null;
  }

  if (
    value.bankrollMode !== "tracked" &&
    value.bankrollMode !== "what-if"
  ) {
    return null;
  }

  if (
    !isNullableNonnegativeFinite(
      value.whatIfBankroll,
    )
  ) {
    return null;
  }

  if (
    !isKellyMultiplier(
      value.kellyMultiplier,
    )
  ) {
    return null;
  }

  return {
    version: BET_SLIP_SIZING_VERSION,
    bankrollMode: value.bankrollMode,
    whatIfBankroll:
      value.whatIfBankroll,
    kellyMultiplier:
      value.kellyMultiplier,
  };
}

export function resolveBetSlipBankroll({
  preference,
  trackedBankroll,
}: {
  preference: BetSlipSizingPreference;
  trackedBankroll: number | null;
}): ResolvedBetSlipBankroll {
  if (
    preference.bankrollMode ===
    "what-if"
  ) {
    if (
      isNonnegativeFinite(
        preference.whatIfBankroll,
      )
    ) {
      return {
        amount:
          preference.whatIfBankroll,
        source: "what-if",
      };
    }

    return {
      amount: null,
      source: "unavailable",
    };
  }

  if (
    isNonnegativeFinite(
      trackedBankroll,
    )
  ) {
    return {
      amount: trackedBankroll,
      source: "tracked",
    };
  }

  return {
    amount: null,
    source: "unavailable",
  };
}

export function updateBetSlipSizingPreference({
  current,
  update,
}: {
  current: BetSlipSizingPreference;
  update: Partial<
    Pick<
      BetSlipSizingPreference,
      | "bankrollMode"
      | "whatIfBankroll"
      | "kellyMultiplier"
    >
  >;
}): BetSlipSizingPreference {
  const candidate: BetSlipSizingPreference =
    {
      ...current,
      ...update,
      version:
        BET_SLIP_SIZING_VERSION,
    };

  return (
    parseBetSlipSizingPreference(
      candidate,
    ) ?? current
  );
}

function isKellyMultiplier(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
  );
}

function isNonnegativeFinite(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0
  );
}

function isNullableNonnegativeFinite(
  value: unknown,
): value is number | null {
  return (
    value === null ||
    isNonnegativeFinite(value)
  );
}

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}
