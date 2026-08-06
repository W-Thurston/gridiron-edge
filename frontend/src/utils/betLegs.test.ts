import { describe, expect, it } from "vitest";
import type { components } from "../api/schema";
import {
  analyzeBetLeg,
  buildGameBetLegId,
  buildPropBetLegId,
  calculateBetMetrics,
  calculatePayout,
  createGameBetLeg,
  createPropBetLeg,
  isPriceAtLeastAsGood,
  modelBreakEvenAmericanOdds,
  parseBetLegsV3,
  parseBetLegV3,
  propSideFromLean,
  type BetLegSource,
  type GameBetLeg,
  type PropBetLeg,
} from "./betLegs";
type EdgeApiRow = components["schemas"]["EdgeRow"];
type PropSummaryApi = components["schemas"]["PropSummary"];

const ADDED_AT = "2026-07-29T15:00:00.000Z";

function edge(overrides: Partial<EdgeApiRow> = {}): EdgeApiRow {
  return {
    provider: "the_odds_api",
    provider_event_id: "event-1",
    sportsbook: "draftkings",
    market_fetched_at: "2026-09-05T12:00:00Z",
    sportsbook_updated_at: "2026-09-05T11:59:00Z",
    commence_time: "2026-09-06T00:20:00Z",
    american_odds: -110,
    away_team: "KC",
    confidence_tier: "High",
    cover_prob: null,
    edge_strength: "strong",
    ev: 0.08,
    game_id: "2026_01_KC_LAC",
    home_team: "LAC",
    kelly_frac: 0.08,
    kelly_stake: 20,
    market_type: "moneyline",
    market_value: 0.45,
    model_key: "random_forest_win_prob",
    model_value: 0.58,
    point_edge: null,
    side: "away",
    ...overrides,
  };
}

function prop(overrides: Partial<PropSummaryApi> = {}): PropSummaryApi {
  return {
    game_id: "2026_01_KC_LAC",
    line_context: {
      confidence_tier: "Moderate",
      lean: "Over",
      line: 274.5,
      p_over: 0.61,
    },
    model_key: "elastic_net_qb_pass_yards",
    player_id: "00-0033873",
    player_name: "Patrick Mahomes",
    position: "QB",
    projection: {
      predicted_mean: 289.4,
      predicted_std: 71.2,
      lo_90: 172.3,
      hi_90: 406.5,
    },
    prop_id: "2026_01_KC_LAC__00-0033873__qb_pass_yards",
    stat_type: "qb_pass_yards",
    team: "KC",
    ...overrides,
  };
}

function gameLeg(
  overrides: Partial<GameBetLeg> = {},
): GameBetLeg {
  const base = createGameBetLeg({
    edge: edge(),
    source: "betslip-edges",
    addedAt: ADDED_AT,
    referenceBankroll: 2500,
    referenceKellyMultiplier: 0.1,
  });
  return { ...base, ...overrides };
}

function propLeg(
  overrides: Partial<PropBetLeg> = {},
): PropBetLeg {
  const base = createPropBetLeg({
    prop: prop(),
    side: "over",
    source: "dashboard-prop-edges",
    addedAt: ADDED_AT,
  });
  return { ...base, ...overrides };
}

describe("canonical wager IDs", () => {
  it("ignores producer source for the same game wager", () => {
    const input = {
      gameId: "2026_01_KC_LAC",
      market: "spread" as const,
      side: "away" as const,
      line: -3.5,
      sportsbook: "draftkings",
    };
    expect(buildGameBetLegId(input)).toBe(
      "game:2026_01_KC_LAC:spread:away:-3.5:draftkings",
    );
    expect(buildGameBetLegId(input)).toBe(buildGameBetLegId(input));
  });

  it("changes when game side or line changes", () => {
    const home = buildGameBetLegId({
      gameId: "game",
      market: "spread",
      side: "home",
      line: -3.5,
      sportsbook: "draftkings",
    });
    const away = buildGameBetLegId({
      gameId: "game",
      market: "spread",
      side: "away",
      line: -3.5,
      sportsbook: "draftkings",
    });
    const moved = buildGameBetLegId({
      gameId: "game",
      market: "spread",
      side: "home",
      line: -4,
      sportsbook: "draftkings",
    });
    expect(new Set([home, away, moved]).size).toBe(3);
  });

  it("canonicalizes numerically equal lines", () => {
    expect(
      buildPropBetLegId({ propId: "prop", side: "over", line: 250 }),
    ).toBe(
      buildPropBetLegId({ propId: "prop", side: "over", line: 250.0 }),
    );
  });

  it("distinguishes prop over and under", () => {
    expect(
      buildPropBetLegId({ propId: "prop", side: "over", line: null }),
    ).not.toBe(
      buildPropBetLegId({ propId: "prop", side: "under", line: null }),
    );
  });
});

describe("createGameBetLeg", () => {
  const create = (
    edgeValue: EdgeApiRow,
    source: BetLegSource = "betslip-edges",
  ) =>
    createGameBetLeg({
      edge: edgeValue,
      source,
      addedAt: ADDED_AT,
      referenceBankroll: 2500,
      referenceKellyMultiplier: 0.1,
    });

  it("keeps moneyline market probability separate from price", () => {
    const leg = create(edge());
    expect(leg.line).toBeNull();
    expect(leg.recommendation.referenceMarketValue).toBe(0.45);
    expect(leg.recommendation.referenceAmericanOdds).toBe(-110);
    expect(leg.recommendation.referenceModelProbability).toBe(0.58);
    expect(leg.draft.currentAmericanOdds).toBe(-110);
    expect(leg.draft.sportsbook).toBe("draftkings");
    expect(leg.recommendation).toMatchObject({
      referenceProvider: "the_odds_api",
      referenceProviderEventId: "event-1",
      referenceSportsbook: "draftkings",
      referenceMarketFetchedAt: "2026-09-05T12:00:00Z",
      referenceSportsbookUpdatedAt: "2026-09-05T11:59:00Z",
      referenceCommenceTime: "2026-09-06T00:20:00Z",
    });
  });

  it("keeps a spread line separate from its selected-side price", () => {
    const leg = create(
      edge({
        market_type: "spread",
        side: "home",
        market_value: -3.5,
        model_value: -6.25,
        cover_prob: 0.59,
        american_odds: -108,
      }),
    );
    expect(leg.line).toBe(-3.5);
    expect(leg.recommendation.referenceAmericanOdds).toBe(-108);
    expect(leg.recommendation.referenceModelProbability).toBe(0.59);
    expect(leg.recommendation.referenceModelValue).toBe(-6.25);
  });

  it("keeps a total line separate from its selected-side price", () => {
    const leg = create(
      edge({
        market_type: "total",
        side: "over",
        market_value: 47.5,
        model_value: 51.2,
        cover_prob: 0.63,
        american_odds: 105,
      }),
    );
    expect(leg.line).toBe(47.5);
    expect(leg.recommendation.referenceAmericanOdds).toBe(105);
    expect(leg.recommendation.referenceModelProbability).toBe(0.63);
  });

  it("retains recommendation sizing provenance", () => {
    const leg = create(edge());
    expect(leg.recommendation).toMatchObject({
      referenceExpectedValue: 0.08,
      referenceEdgeStrength: "strong",
      referenceKellyFraction: 0.08,
      referenceKellyStake: 20,
      referenceBankroll: 2500,
      referenceKellyMultiplier: 0.1,
    });
  });

  it("uses a canonical ID independent of producer source", () => {
    expect(create(edge(), "dashboard-featured").id).toBe(
      create(edge(), "game-detail-lean").id,
    );
  });


  it("distinguishes otherwise identical sportsbook offers", () => {
    expect(create(edge({ sportsbook: "draftkings" })).id).not.toBe(
      create(edge({ sportsbook: "fanduel" })).id,
    );
    expect(create(edge({ sportsbook: null })).id).toContain(":consensus");
  });

  it("rejects unsupported market and side combinations", () => {
    expect(() => create(edge({ market_type: "prop" }))).toThrow();
    expect(() => create(edge({ market_type: "total", side: "home" }))).toThrow();
  });

  it("rejects zero or fractional American prices", () => {
    expect(() => create(edge({ american_odds: 0 }))).toThrow();
    expect(() => create(edge({ american_odds: -110.5 }))).toThrow();
  });
});

describe("createPropBetLeg", () => {
  it("uses the real game ID and preserves projection context", () => {
    const leg = propLeg();
    expect(leg.gameId).toBe("2026_01_KC_LAC");
    expect(leg.gameId).not.toBe(leg.propId);
    expect(leg.predictedMean).toBe(289.4);
    expect(leg.line).toBe(274.5);
  });

  it("derives over and under model probabilities", () => {
    const over = createPropBetLeg({
      prop: prop(),
      side: "over",
      source: "dashboard-prop-edges",
      addedAt: ADDED_AT,
    });
    const under = createPropBetLeg({
      prop: prop(),
      side: "under",
      source: "player-prop",
      addedAt: ADDED_AT,
    });
    expect(over.recommendation.referenceModelProbability).toBe(0.61);
    expect(under.recommendation.referenceModelProbability).toBeCloseTo(0.39);
  });

  it("keeps probability null when p_over is unavailable", () => {
    const leg = createPropBetLeg({
      prop: prop({ line_context: null }),
      side: "over",
      source: "player-prop",
      addedAt: ADDED_AT,
    });
    expect(leg.line).toBeNull();
    expect(leg.recommendation.referenceModelProbability).toBeNull();
  });

  it("never fabricates a reference or current price", () => {
    const leg = propLeg();
    expect(leg.recommendation.referenceAmericanOdds).toBeNull();
    expect(leg.draft.currentAmericanOdds).toBeNull();
  });

  it("deduplicates the same prop across producers", () => {
    const dashboard = createPropBetLeg({
      prop: prop(),
      side: "over",
      source: "dashboard-prop-edges",
      addedAt: ADDED_AT,
    });
    const detail = createPropBetLeg({
      prop: prop(),
      side: "over",
      source: "player-prop",
      addedAt: ADDED_AT,
    });
    expect(dashboard.id).toBe(detail.id);
  });
});

describe("bet calculations", () => {
  it("calculates metrics for negative American odds", () => {
    const result = calculateBetMetrics({
      americanOdds: -110,
      modelProbability: 0.58,
    });
    expect(result?.decimalOdds).toBeCloseTo(1.9090909);
    expect(result?.impliedProbability).toBeCloseTo(0.5238095);
    expect(result?.expectedValue).toBeCloseTo(0.1072727);
    expect(result?.fullKellyFraction).toBeCloseTo(0.118);
  });

  it("calculates metrics for positive American odds", () => {
    const result = calculateBetMetrics({
      americanOdds: 150,
      modelProbability: 0.45,
    });
    expect(result?.decimalOdds).toBe(2.5);
    expect(result?.impliedProbability).toBe(0.4);
    expect(result?.expectedValue).toBeCloseTo(0.125);
    expect(result?.fullKellyFraction).toBeCloseTo(0.0833333);
  });

  it("returns price-only metrics without model probability", () => {
    expect(
      calculateBetMetrics({ americanOdds: -110, modelProbability: null }),
    ).toEqual({
      decimalOdds: expect.any(Number),
      impliedProbability: expect.any(Number),
      expectedValue: null,
      fullKellyFraction: null,
    });
  });

  it("rejects missing or invalid price", () => {
    expect(
      calculateBetMetrics({ americanOdds: null, modelProbability: 0.6 }),
    ).toBeNull();
    expect(
      calculateBetMetrics({ americanOdds: 0, modelProbability: 0.6 }),
    ).toBeNull();
  });

  it("floors negative Kelly at zero", () => {
    expect(
      calculateBetMetrics({ americanOdds: -150, modelProbability: 0.3 })
        ?.fullKellyFraction,
    ).toBe(0);
  });

  it("calculates payout and profit for negative and positive prices", () => {
    expect(calculatePayout({ americanOdds: -110, stake: 110 })).toEqual({
      payout: 210,
      profit: 100,
    });
    expect(calculatePayout({ americanOdds: 150, stake: 100 })).toEqual({
      payout: 250,
      profit: 150,
    });
  });

  it("preserves zero stake and rejects negative stake", () => {
    expect(calculatePayout({ americanOdds: -110, stake: 0 })).toEqual({
      payout: 0,
      profit: 0,
    });
    expect(calculatePayout({ americanOdds: -110, stake: -1 })).toBeNull();
  });

  it("calculates favored and underdog break-even prices", () => {
    expect(modelBreakEvenAmericanOdds(0.65)).toBe(-186);
    expect(modelBreakEvenAmericanOdds(0.35)).toBe(186);
    expect(modelBreakEvenAmericanOdds(null)).toBeNull();
  });

  it("compares price quality using decimal economics", () => {
    expect(isPriceAtLeastAsGood(-115, -124)).toBe(true);
    expect(isPriceAtLeastAsGood(-130, -124)).toBe(false);
    expect(isPriceAtLeastAsGood(130, 120)).toBe(true);
    expect(isPriceAtLeastAsGood(110, 120)).toBe(false);
  });
});

describe("analyzeBetLeg", () => {
  it("keeps reference and current calculations separate", () => {
    const leg = gameLeg({
      draft: {
        ...gameLeg().draft,
        currentAmericanOdds: -125,
      },
    });

    const analysis = analyzeBetLeg({
      leg,
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.reference
        ?.decimalOdds,
    ).toBeCloseTo(1.9090909);

    expect(
      analysis.current
        ?.decimalOdds,
    ).toBeCloseTo(1.8);

    expect(
      analysis.reference
        ?.expectedValue,
    ).toBeCloseTo(0.1072727);

    expect(
      analysis.current
        ?.expectedValue,
    ).toBeCloseTo(0.044);
  });

  it("calculates the model break-even price", () => {
    const analysis = analyzeBetLeg({
      leg: gameLeg(),
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.breakEvenAmericanOdds,
    ).toBe(-138);
  });

  it("identifies whether the current price remains acceptable", () => {
    const acceptable = analyzeBetLeg({
      leg: gameLeg({
        draft: {
          ...gameLeg().draft,
          currentAmericanOdds: -125,
        },
      }),
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    const unacceptable = analyzeBetLeg({
      leg: gameLeg({
        draft: {
          ...gameLeg().draft,
          currentAmericanOdds: -150,
        },
      }),
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      acceptable.currentPriceIsAcceptable,
    ).toBe(true);

    expect(
      unacceptable.currentPriceIsAcceptable,
    ).toBe(false);
  });

  it("calculates multiplier-adjusted Kelly dollars", () => {
    const analysis = analyzeBetLeg({
      leg: gameLeg(),
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.current
        ?.fullKellyFraction,
    ).toBeCloseTo(0.118);

    expect(
      analysis.suggestedStake,
    ).toBeCloseTo(29.5);
  });

  it("keeps proposed stake separate from suggested stake", () => {
    const leg = gameLeg({
      draft: {
        ...gameLeg().draft,
        proposedStake: 50,
      },
    });

    const analysis = analyzeBetLeg({
      leg,
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.suggestedStake,
    ).toBeCloseTo(29.5);

    expect(
      analysis.payout,
    ).toBeCloseTo(95.454545);

    expect(
      analysis.profit,
    ).toBeCloseTo(45.454545);
  });

  it("blocks price-dependent calculations when current price is missing", () => {
    const leg = gameLeg({
      draft: {
        ...gameLeg().draft,
        currentAmericanOdds: null,
        proposedStake: 50,
      },
    });

    const analysis = analyzeBetLeg({
      leg,
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.reference,
    ).not.toBeNull();

    expect(
      analysis.current,
    ).toBeNull();

    expect(
      analysis.currentPriceIsAcceptable,
    ).toBeNull();

    expect(
      analysis.suggestedStake,
    ).toBeNull();

    expect(
      analysis.payout,
    ).toBeNull();

    expect(
      analysis.profit,
    ).toBeNull();
  });

  it("blocks EV and Kelly when model probability is missing", () => {
    const analysis = analyzeBetLeg({
      leg: propLeg({
        draft: {
          ...propLeg().draft,
          currentAmericanOdds: -110,
          proposedStake: 25,
        },
        recommendation: {
          ...propLeg().recommendation,
          referenceModelProbability:
            null,
        },
      }),
      bankroll: 2500,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.current
        ?.decimalOdds,
    ).toBeCloseTo(1.9090909);

    expect(
      analysis.current
        ?.impliedProbability,
    ).toBeCloseTo(0.5238095);

    expect(
      analysis.current
        ?.expectedValue,
    ).toBeNull();

    expect(
      analysis.current
        ?.fullKellyFraction,
    ).toBeNull();

    expect(
      analysis.breakEvenAmericanOdds,
    ).toBeNull();

    expect(
      analysis.suggestedStake,
    ).toBeNull();

    expect(
      analysis.payout,
    ).toBeCloseTo(47.7272727);
  });

  it("blocks dollar sizing without an explicit bankroll", () => {
    const analysis = analyzeBetLeg({
      leg: gameLeg(),
      bankroll: null,
      kellyMultiplier: 0.1,
    });

    expect(
      analysis.current
        ?.fullKellyFraction,
    ).not.toBeNull();

    expect(
      analysis.suggestedStake,
    ).toBeNull();
  });

  it("blocks dollar sizing without an explicit Kelly multiplier", () => {
    const analysis = analyzeBetLeg({
      leg: gameLeg(),
      bankroll: 2500,
      kellyMultiplier: null,
    });

    expect(
      analysis.current
        ?.fullKellyFraction,
    ).not.toBeNull();

    expect(
      analysis.suggestedStake,
    ).toBeNull();
  });

  it("preserves zero bankroll and zero multiplier as valid inputs", () => {
    expect(
      analyzeBetLeg({
        leg: gameLeg(),
        bankroll: 0,
        kellyMultiplier: 0.1,
      }).suggestedStake,
    ).toBe(0);

    expect(
      analyzeBetLeg({
        leg: gameLeg(),
        bankroll: 2500,
        kellyMultiplier: 0,
      }).suggestedStake,
    ).toBe(0);
  });
});

describe("v3 runtime parsing", () => {
  it("accepts valid game and prop legs", () => {
    expect(parseBetLegV3(gameLeg())).toEqual(gameLeg());
    expect(parseBetLegV3(propLeg())).toEqual(propLeg());
  });

  it("rejects retired versions and malformed sources", () => {
    expect(parseBetLegV3({ ...gameLeg(), version: 2 })).toBeNull();
    expect(parseBetLegV3({ ...gameLeg(), source: "unknown" })).toBeNull();
  });

  it("rejects invalid market-side combinations", () => {
    expect(
      parseBetLegV3({ ...gameLeg(), market: "total", side: "home" }),
    ).toBeNull();
  });

  it("rejects a noncanonical ID", () => {
    expect(parseBetLegV3({ ...gameLeg(), id: "producer-specific" })).toBeNull();
  });

  it("rejects zero, NaN, and infinite prices", () => {
    for (const currentAmericanOdds of [0, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(
        parseBetLegV3({
          ...gameLeg(),
          draft: { ...gameLeg().draft, currentAmericanOdds },
        }),
      ).toBeNull();
    }
  });

  it("rejects malformed nested blocks", () => {
    expect(parseBetLegV3({ ...gameLeg(), recommendation: null })).toBeNull();
    expect(parseBetLegV3({ ...gameLeg(), draft: {} })).toBeNull();
  });

  it("salvages valid legs from a mixed array", () => {
    expect(
      parseBetLegsV3([gameLeg(), { ...propLeg(), version: 1 }, propLeg()]),
    ).toEqual([gameLeg(), propLeg()]);
  });

  it("returns an empty list for non-array input", () => {
    expect(parseBetLegsV3(gameLeg())).toEqual([]);
  });
});

describe("propSideFromLean", () => {
  it("normalizes Over and Under", () => {
    expect(
      propSideFromLean("Over"),
    ).toBe("over");

    expect(
      propSideFromLean(" under "),
    ).toBe("under");
  });

  it("rejects missing and no-edge leans", () => {
    expect(
      propSideFromLean(null),
    ).toBeNull();

    expect(
      propSideFromLean("No Edge"),
    ).toBeNull();
  });
});
