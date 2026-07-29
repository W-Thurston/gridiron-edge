import type {
  BetLeg,
} from "./betLegs";
import {
  calculatePayout,
} from "./betLegs";
import {
  americanToDecimal,
  decimalToAmerican,
} from "./odds";

export type BetSlipSummaryMode =
  | "single"
  | "parlay";

export type IncompleteBetSlipReason =
  | "no_legs"
  | "missing_current_price"
  | "missing_proposed_stake"
  | "missing_parlay_stake";

export type SingleBetSlipSummary = {
  mode: "single";
  isComplete: boolean;
  incompleteReasons:
    IncompleteBetSlipReason[];
  legCount: number;
  totalStake: number | null;
  potentialPayout: number | null;
  potentialProfit: number | null;
};

export type ParlayBetSlipSummary = {
  mode: "parlay";
  isComplete: boolean;
  incompleteReasons:
    IncompleteBetSlipReason[];
  legCount: number;
  parlayStake: number | null;
  combinedDecimalOdds: number | null;
  combinedAmericanOdds: number | null;
  potentialPayout: number | null;
  potentialProfit: number | null;
  combinedModelProbability: null;
  combinedExpectedValue: null;
  combinedKellyFraction: null;
};

export type BetSlipSummary =
  | SingleBetSlipSummary
  | ParlayBetSlipSummary;

export function summarizeBetSlip({
  legs,
  mode,
  parlayStake,
}: {
  legs: BetLeg[];
  mode: BetSlipSummaryMode;
  parlayStake: number | null;
}): BetSlipSummary {
  if (mode === "parlay") {
    return summarizeParlay({
      legs,
      parlayStake,
    });
  }

  return summarizeSingles(legs);
}

function summarizeSingles(
  legs: BetLeg[],
): SingleBetSlipSummary {
  const incompleteReasons =
    singleIncompleteReasons(legs);

  if (
    legs.length === 0 ||
    incompleteReasons.length > 0
  ) {
    return {
      mode: "single",
      isComplete: false,
      incompleteReasons,
      legCount: legs.length,
      totalStake: null,
      potentialPayout: null,
      potentialProfit: null,
    };
  }

  let totalStake = 0;
  let potentialPayout = 0;
  let potentialProfit = 0;

  for (const leg of legs) {
    const proposedStake =
      leg.draft.proposedStake;
    const currentAmericanOdds =
      leg.draft.currentAmericanOdds;

    if (
      proposedStake == null ||
      currentAmericanOdds == null
    ) {
      return {
        mode: "single",
        isComplete: false,
        incompleteReasons:
          singleIncompleteReasons(legs),
        legCount: legs.length,
        totalStake: null,
        potentialPayout: null,
        potentialProfit: null,
      };
    }

    const payout = calculatePayout({
      americanOdds:
        currentAmericanOdds,
      stake: proposedStake,
    });

    if (payout == null) {
      return {
        mode: "single",
        isComplete: false,
        incompleteReasons:
          singleIncompleteReasons(legs),
        legCount: legs.length,
        totalStake: null,
        potentialPayout: null,
        potentialProfit: null,
      };
    }

    totalStake += proposedStake;
    potentialPayout += payout.payout;
    potentialProfit += payout.profit;
  }

  return {
    mode: "single",
    isComplete: true,
    incompleteReasons: [],
    legCount: legs.length,
    totalStake,
    potentialPayout,
    potentialProfit,
  };
}

function summarizeParlay({
  legs,
  parlayStake,
}: {
  legs: BetLeg[];
  parlayStake: number | null;
}): ParlayBetSlipSummary {
  const incompleteReasons =
    parlayIncompleteReasons({
      legs,
      parlayStake,
    });

  if (
    legs.length === 0 ||
    incompleteReasons.length > 0
  ) {
    return {
      mode: "parlay",
      isComplete: false,
      incompleteReasons,
      legCount: legs.length,
      parlayStake,
      combinedDecimalOdds: null,
      combinedAmericanOdds: null,
      potentialPayout: null,
      potentialProfit: null,
      combinedModelProbability: null,
      combinedExpectedValue: null,
      combinedKellyFraction: null,
    };
  }

  const combinedDecimalOdds =
    legs.reduce(
      (product, leg) =>
        product *
        americanToDecimal(
          leg.draft
            .currentAmericanOdds as number,
        ),
      1,
    );

  const stake = parlayStake as number;
  const potentialPayout =
    stake * combinedDecimalOdds;
  const potentialProfit =
    potentialPayout - stake;

  return {
    mode: "parlay",
    isComplete: true,
    incompleteReasons: [],
    legCount: legs.length,
    parlayStake: stake,
    combinedDecimalOdds,
    combinedAmericanOdds:
      decimalToAmerican(
        combinedDecimalOdds,
      ),
    potentialPayout,
    potentialProfit,
    combinedModelProbability: null,
    combinedExpectedValue: null,
    combinedKellyFraction: null,
  };
}

function singleIncompleteReasons(
  legs: BetLeg[],
): IncompleteBetSlipReason[] {
  const reasons =
    new Set<IncompleteBetSlipReason>();

  if (legs.length === 0) {
    reasons.add("no_legs");
  }

  for (const leg of legs) {
    if (
      leg.draft.currentAmericanOdds ==
      null
    ) {
      reasons.add(
        "missing_current_price",
      );
    }

    if (
      leg.draft.proposedStake == null
    ) {
      reasons.add(
        "missing_proposed_stake",
      );
    }
  }

  return [...reasons];
}

function parlayIncompleteReasons({
  legs,
  parlayStake,
}: {
  legs: BetLeg[];
  parlayStake: number | null;
}): IncompleteBetSlipReason[] {
  const reasons =
    new Set<IncompleteBetSlipReason>();

  if (legs.length === 0) {
    reasons.add("no_legs");
  }

  for (const leg of legs) {
    if (
      leg.draft.currentAmericanOdds ==
      null
    ) {
      reasons.add(
        "missing_current_price",
      );
    }
  }

  if (parlayStake == null) {
    reasons.add(
      "missing_parlay_stake",
    );
  }

  return [...reasons];
}
