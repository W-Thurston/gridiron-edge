import {
  describe,
  expect,
  it,
} from "vitest";
import type {
  components,
} from "../api/schema";
import {
  createGameBetLeg,
  createPropBetLeg,
  type BetLeg,
} from "./betLegs";
import {
  summarizeBetSlip,
} from "./betSlipSummary";

type EdgeApiRow =
  components["schemas"]["EdgeRow"];

type PropApi =
  components["schemas"]["PropSummary"];

const ADDED_AT =
  "2026-07-29T21:00:00.000Z";

function edge(
  overrides: Partial<EdgeApiRow> = {},
): EdgeApiRow {
  return {
    american_odds: -110,
    away_team: "KC",
    cover_prob: null,
    edge_strength: "strong",
    ev: 0.08,
    game_id:
      "2026_01_KC_LAC",
    home_team: "LAC",
    kelly_frac: 0.08,
    kelly_stake: 20,
    market_type: "moneyline",
    market_value: 0.45,
    model_key:
      "random_forest_win_prob",
    model_value: 0.58,
    point_edge: null,
    side: "away",
    ...overrides,
  };
}

function prop(): PropApi {
  return {
    game_id:
      "2026_01_KC_LAC",
    line_context: {
      line: 274.5,
      p_over: 0.61,
      lean: "Over",
      confidence_tier:
        "Moderate",
    },
    model_key:
      "elastic_net_qb_pass_yards",
    player_id: "player-1",
    player_name:
      "Patrick Mahomes",
    position: "QB",
    projection: {
      predicted_mean: 289.4,
      predicted_std: 71.2,
    },
    prop_id:
      "2026_01_KC_LAC__player-1__qb_pass_yards",
    stat_type:
      "qb_pass_yards",
    team: "KC",
  };
}

function pricedGameLeg({
  americanOdds = -110,
  proposedStake = 110,
}: {
  americanOdds?: number;
  proposedStake?: number;
} = {}): BetLeg {
  const leg = createGameBetLeg({
    edge: edge({
      american_odds:
        americanOdds,
    }),
    source: "betslip-edges",
    addedAt: ADDED_AT,
    referenceBankroll: 2500,
    referenceKellyMultiplier:
      0.1,
  });

  return {
    ...leg,
    draft: {
      ...leg.draft,
      currentAmericanOdds:
        americanOdds,
      proposedStake,
    },
  };
}

function unpricedPropLeg(): BetLeg {
  return createPropBetLeg({
    prop: prop(),
    side: "over",
    source:
      "dashboard-prop-edges",
    addedAt: ADDED_AT,
  });
}

describe("summarizeBetSlip singles", () => {
  it(
    "returns incomplete for an empty slip",
    () => {
      expect(
        summarizeBetSlip({
          legs: [],
          mode: "single",
          parlayStake: null,
        }),
      ).toEqual({
        mode: "single",
        isComplete: false,
        incompleteReasons: [
          "no_legs",
        ],
        legCount: 0,
        totalStake: null,
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );

  it(
    "calculates one complete single",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
          ],
          mode: "single",
          parlayStake: null,
        });

      expect(summary).toEqual({
        mode: "single",
        isComplete: true,
        incompleteReasons: [],
        legCount: 1,
        totalStake: 110,
        potentialPayout: 210,
        potentialProfit: 100,
      });
    },
  );

  it(
    "sums , and profits",
    () => {
        const summary =
        summarizeBetSlip({
            legs: [
            pricedGameLeg({
                americanOdds: -110,
                proposedStake: 110,
            }),
            pricedGameLeg({
                americanOdds: 150,
                proposedStake: 100,
            }),
            ],
            mode: "single",
            parlayStake: null,
        });

        expect(summary.mode).toBe(
        "single",
        );

        if (
        summary.mode !== "single"
        ) {
        throw new Error(
            "Expected single summary",
        );
        }

        expect(
        summary.totalStake,
        ).toBe(210);

        expect(
        summary.potentialPayout,
        ).toBe(460);

        expect(
        summary.potentialProfit,
        ).toBe(250);
    },
    );

  it(
    "preserves zero proposed stake",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg({
              proposedStake: 0,
            }),
          ],
          mode: "single",
          parlayStake: null,
        });

      expect(summary).toMatchObject({
        isComplete: true,
        totalStake: 0,
        potentialPayout: 0,
        potentialProfit: 0,
      });
    },
  );

  it(
    "blocks all aggregate economics when a leg lacks current price",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
            unpricedPropLeg(),
          ],
          mode: "single",
          parlayStake: null,
        });

      expect(summary).toMatchObject({
        isComplete: false,
        incompleteReasons: [
          "missing_current_price",
          "missing_proposed_stake",
        ],
        totalStake: null,
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );

  it(
    "blocks all aggregate economics when a leg lacks proposed stake",
    () => {
      const incompleteLeg = {
        ...pricedGameLeg(),
        draft: {
          ...pricedGameLeg().draft,
          proposedStake: null,
        },
      };

      const summary =
        summarizeBetSlip({
          legs: [incompleteLeg],
          mode: "single",
          parlayStake: null,
        });

      expect(summary).toMatchObject({
        isComplete: false,
        incompleteReasons: [
          "missing_proposed_stake",
        ],
        totalStake: null,
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );
});

describe("summarizeBetSlip parlay", () => {
  it(
    "returns incomplete for an empty parlay",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [],
          mode: "parlay",
          parlayStake: null,
        });

      expect(summary).toMatchObject({
        mode: "parlay",
        isComplete: false,
        incompleteReasons: [
          "no_legs",
          "missing_parlay_stake",
        ],
        combinedDecimalOdds: null,
        combinedAmericanOdds: null,
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );

  it(
    "calculates quoted combined odds, payout, and profit",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg({
              americanOdds: -110,
            }),
            pricedGameLeg({
              americanOdds: 150,
            }),
          ],
          mode: "parlay",
          parlayStake: 100,
        });

      expect(summary.mode).toBe(
        "parlay",
      );

      if (
        summary.mode !== "parlay"
      ) {
        throw new Error(
          "Expected parlay summary",
        );
      }

      expect(
        summary.isComplete,
      ).toBe(true);

      expect(
        summary.combinedDecimalOdds,
      ).toBeCloseTo(
        4.7727272727,
      );

      expect(
        summary.combinedAmericanOdds,
      ).toBe(377);

      expect(
        summary.potentialPayout,
      ).toBeCloseTo(
        477.27272727,
      );

      expect(
        summary.potentialProfit,
      ).toBeCloseTo(
        377.27272727,
      );
    },
  );

  it(
    "ignores per-leg stakes in parlay mode",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg({
              proposedStake: 500,
            }),
            pricedGameLeg({
              americanOdds: 150,
              proposedStake: 900,
            }),
          ],
          mode: "parlay",
          parlayStake: 10,
        });

      expect(summary).toMatchObject({
        mode: "parlay",
        isComplete: true,
        parlayStake: 10,
      });

      expect(
        summary.potentialPayout,
      ).toBeCloseTo(
        47.7272727,
      );
    },
  );

  it(
    "preserves zero as a valid parlay stake",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
            pricedGameLeg({
              americanOdds: 150,
            }),
          ],
          mode: "parlay",
          parlayStake: 0,
        });

      expect(summary).toMatchObject({
        isComplete: true,
        parlayStake: 0,
        potentialPayout: 0,
        potentialProfit: 0,
      });
    },
  );

  it(
    "blocks combined payout when any leg lacks current price",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
            unpricedPropLeg(),
          ],
          mode: "parlay",
          parlayStake: 25,
        });

      expect(summary).toMatchObject({
        isComplete: false,
        incompleteReasons: [
          "missing_current_price",
        ],
        combinedDecimalOdds: null,
        combinedAmericanOdds: null,
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );

  it(
    "blocks combined payout when parlay stake is missing",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
            pricedGameLeg({
              americanOdds: 150,
            }),
          ],
          mode: "parlay",
          parlayStake: null,
        });

      expect(summary).toMatchObject({
        isComplete: false,
        incompleteReasons: [
          "missing_parlay_stake",
        ],
        potentialPayout: null,
        potentialProfit: null,
      });
    },
  );

  it(
    "never produces combined model probability, EV, or Kelly",
    () => {
      const summary =
        summarizeBetSlip({
          legs: [
            pricedGameLeg(),
            pricedGameLeg({
              americanOdds: 150,
            }),
          ],
          mode: "parlay",
          parlayStake: 25,
        });

      expect(summary).toMatchObject({
        combinedModelProbability:
          null,
        combinedExpectedValue: null,
        combinedKellyFraction: null,
      });
    },
  );
});
