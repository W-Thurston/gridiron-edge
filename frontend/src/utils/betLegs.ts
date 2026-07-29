import type { components } from "../api/schema";
import {
  americanToDecimal,
  decimalToAmerican,
  impliedProb,
  kelly,
} from "./odds";

type EdgeApiRow = components["schemas"]["EdgeRow"];
type PropSummaryApi = components["schemas"]["PropSummary"];
type PropDetailApi = components["schemas"]["PropDetail"];
type PropApi = PropSummaryApi | PropDetailApi;

export const BET_LEG_VERSION = 2 as const;

export type BetLegSource =
  | "betslip-edges"
  | "dashboard-featured"
  | "dashboard-model-edges"
  | "dashboard-prop-edges"
  | "game-detail-lean"
  | "game-detail-prop"
  | "player-prop";

export type GameMarket = "moneyline" | "spread" | "total";
export type GameSide = "home" | "away" | "over" | "under";
export type PropSide = "over" | "under";

export type RecommendationSnapshot = {
  modelKey: string;
  referenceAmericanOdds: number | null;
  referenceModelProbability: number | null;
  referenceModelValue: number | null;
  referenceMarketValue: number | null;
  referenceExpectedValue: number | null;
  referenceEdgeStrength: string | null;
  referenceKellyFraction: number | null;
  referenceKellyStake: number | null;
  referenceBankroll: number | null;
  referenceKellyMultiplier: number | null;
};

export type BetDraftInputs = {
  currentAmericanOdds: number | null;
  proposedStake: number | null;
  sportsbook: string | null;
  note: string | null;
};

type BetLegBase = {
  version: typeof BET_LEG_VERSION;
  id: string;
  source: BetLegSource;
  addedAt: string;
  recommendation: RecommendationSnapshot;
  draft: BetDraftInputs;
};

export type GameBetLeg = BetLegBase & {
  kind: "game";
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  market: GameMarket;
  side: GameSide;
  line: number | null;
};

export type PropBetLeg = BetLegBase & {
  kind: "prop";
  propId: string;
  gameId: string;
  playerId: string;
  playerName: string;
  position: string;
  team: string;
  statType: string;
  side: PropSide;
  line: number | null;
  predictedMean: number | null;
};

export type BetLeg = GameBetLeg | PropBetLeg;

export type BetCalculation = {
  decimalOdds: number;
  impliedProbability: number;
  expectedValue: number | null;
  fullKellyFraction: number | null;
};

const SOURCES = new Set<BetLegSource>([
  "betslip-edges",
  "dashboard-featured",
  "dashboard-model-edges",
  "dashboard-prop-edges",
  "game-detail-lean",
  "game-detail-prop",
  "player-prop",
]);

const GAME_MARKETS = new Set<GameMarket>([
  "moneyline",
  "spread",
  "total",
]);

const PROP_SIDES = new Set<PropSide>(["over", "under"]);

export function buildGameBetLegId({
  gameId,
  market,
  side,
  line,
}: {
  gameId: string;
  market: GameMarket;
  side: GameSide;
  line: number | null;
}): string {
  return ["game", gameId, market, side, lineToken(line)].join(":");
}

export function buildPropBetLegId({
  propId,
  side,
  line,
}: {
  propId: string;
  side: PropSide;
  line: number | null;
}): string {
  return ["prop", propId, side, lineToken(line)].join(":");
}

export function createGameBetLeg({
  edge,
  source,
  addedAt,
  referenceBankroll,
  referenceKellyMultiplier,
}: {
  edge: EdgeApiRow;
  source: BetLegSource;
  addedAt: string;
  referenceBankroll: number | null;
  referenceKellyMultiplier: number | null;
}): GameBetLeg {
  const market = parseGameMarket(edge.market_type);
  const side = parseGameSide(edge.side, market);
  const line =
    market === "spread" || market === "total"
      ? finiteOrNull(edge.market_value)
      : null;
  const referenceAmericanOdds = parseAmericanOdds(edge.american_odds);

  if (!referenceAmericanOdds) {
    throw new Error("Edge american_odds must be a nonzero integer");
  }

  const referenceModelProbability =
    market === "moneyline"
      ? probabilityOrNull(edge.model_value)
      : probabilityOrNull(edge.cover_prob);

  return {
    version: BET_LEG_VERSION,
    kind: "game",
    id: buildGameBetLegId({
      gameId: edge.game_id,
      market,
      side,
      line,
    }),
    source,
    addedAt,
    gameId: edge.game_id,
    awayTeam: edge.away_team,
    homeTeam: edge.home_team,
    market,
    side,
    line,
    recommendation: {
      modelKey: edge.model_key,
      referenceAmericanOdds,
      referenceModelProbability,
      referenceModelValue: finiteOrNull(edge.model_value),
      referenceMarketValue: finiteOrNull(edge.market_value),
      referenceExpectedValue: finiteOrNull(edge.ev),
      referenceEdgeStrength: edge.edge_strength || null,
      referenceKellyFraction: finiteOrNull(edge.kelly_frac),
      referenceKellyStake: nonnegativeOrNull(edge.kelly_stake),
      referenceBankroll: nonnegativeOrNull(referenceBankroll),
      referenceKellyMultiplier: nonnegativeOrNull(
        referenceKellyMultiplier,
      ),
    },
    draft: {
      currentAmericanOdds: referenceAmericanOdds,
      proposedStake: null,
      sportsbook: null,
      note: null,
    },
  };
}

export function createPropBetLeg({
  prop,
  side,
  source,
  addedAt,
}: {
  prop: PropApi;
  side: PropSide;
  source: BetLegSource;
  addedAt: string;
}): PropBetLeg {
  const line = finiteOrNull(prop.line_context?.line);
  const pOver = probabilityOrNull(prop.line_context?.p_over);
  const referenceModelProbability =
    pOver == null ? null : side === "over" ? pOver : 1 - pOver;

  return {
    version: BET_LEG_VERSION,
    kind: "prop",
    id: buildPropBetLegId({ propId: prop.prop_id, side, line }),
    source,
    addedAt,
    propId: prop.prop_id,
    gameId: prop.game_id,
    playerId: prop.player_id,
    playerName: prop.player_name,
    position: prop.position,
    team: prop.team,
    statType: prop.stat_type,
    side,
    line,
    predictedMean: finiteOrNull(prop.projection?.predicted_mean),
    recommendation: {
      modelKey: prop.model_key,
      referenceAmericanOdds: null,
      referenceModelProbability,
      referenceModelValue: finiteOrNull(prop.projection?.predicted_mean),
      referenceMarketValue: line,
      referenceExpectedValue: null,
      referenceEdgeStrength: null,
      referenceKellyFraction: null,
      referenceKellyStake: null,
      referenceBankroll: null,
      referenceKellyMultiplier: null,
    },
    draft: {
      currentAmericanOdds: null,
      proposedStake: null,
      sportsbook: null,
      note: null,
    },
  };
}

export function calculateBetMetrics({
  americanOdds,
  modelProbability,
}: {
  americanOdds: number | null;
  modelProbability: number | null;
}): BetCalculation | null {
  const validOdds = parseAmericanOdds(americanOdds);
  if (validOdds == null) return null;

  const decimalOdds = americanToDecimal(validOdds);
  const impliedProbability = impliedProb(validOdds);
  const probability = probabilityOrNull(modelProbability);

  if (probability == null) {
    return {
      decimalOdds,
      impliedProbability,
      expectedValue: null,
      fullKellyFraction: null,
    };
  }

  return {
    decimalOdds,
    impliedProbability,
    expectedValue: probability * decimalOdds - 1,
    fullKellyFraction: kelly(probability, decimalOdds - 1),
  };
}

export function calculatePayout({
  americanOdds,
  stake,
}: {
  americanOdds: number | null;
  stake: number | null;
}): { payout: number; profit: number } | null {
  const validOdds = parseAmericanOdds(americanOdds);
  const validStake = nonnegativeOrNull(stake);
  if (validOdds == null || validStake == null) return null;

  const payout = validStake * americanToDecimal(validOdds);
  return { payout, profit: payout - validStake };
}

export function modelBreakEvenAmericanOdds(
  modelProbability: number | null,
): number | null {
  const probability = probabilityOrNull(modelProbability);
  if (probability == null) return null;
  return decimalToAmerican(1 / probability);
}

export function isPriceAtLeastAsGood(
  currentAmericanOdds: number,
  thresholdAmericanOdds: number,
): boolean {
  const current = parseAmericanOdds(currentAmericanOdds);
  const threshold = parseAmericanOdds(thresholdAmericanOdds);
  if (current == null || threshold == null) return false;
  return americanToDecimal(current) >= americanToDecimal(threshold);
}

export function parseBetLegV2(value: unknown): BetLeg | null {
  if (!isRecord(value) || value.version !== BET_LEG_VERSION) return null;
  if (!isSource(value.source)) return null;
  if (!isNonemptyString(value.id) || !isNonemptyString(value.addedAt)) {
    return null;
  }

  const recommendation = parseRecommendation(value.recommendation);
  const draft = parseDraft(value.draft);
  if (!recommendation || !draft) return null;

  if (value.kind === "game") {
    const market = parseGameMarketOrNull(value.market);
    if (!market || !isGameSideForMarket(value.side, market)) return null;
    if (
      !isNonemptyString(value.gameId) ||
      !isNonemptyString(value.awayTeam) ||
      !isNonemptyString(value.homeTeam) ||
      !isNullableFinite(value.line)
    ) {
      return null;
    }

    const line = value.line as number | null;
    const expectedId = buildGameBetLegId({
      gameId: value.gameId,
      market,
      side: value.side,
      line,
    });
    if (value.id !== expectedId) return null;

    return {
      version: BET_LEG_VERSION,
      kind: "game",
      id: value.id,
      source: value.source,
      addedAt: value.addedAt,
      gameId: value.gameId,
      awayTeam: value.awayTeam,
      homeTeam: value.homeTeam,
      market,
      side: value.side,
      line,
      recommendation,
      draft,
    };
  }

  if (value.kind === "prop") {
    if (
      !isNonemptyString(value.propId) ||
      !isNonemptyString(value.gameId) ||
      !isNonemptyString(value.playerId) ||
      !isNonemptyString(value.playerName) ||
      !isNonemptyString(value.position) ||
      !isNonemptyString(value.team) ||
      !isNonemptyString(value.statType) ||
      !isPropSide(value.side) ||
      !isNullableFinite(value.line) ||
      !isNullableFinite(value.predictedMean)
    ) {
      return null;
    }

    const line = value.line as number | null;
    const expectedId = buildPropBetLegId({
      propId: value.propId,
      side: value.side,
      line,
    });
    if (value.id !== expectedId) return null;

    return {
      version: BET_LEG_VERSION,
      kind: "prop",
      id: value.id,
      source: value.source,
      addedAt: value.addedAt,
      propId: value.propId,
      gameId: value.gameId,
      playerId: value.playerId,
      playerName: value.playerName,
      position: value.position,
      team: value.team,
      statType: value.statType,
      side: value.side,
      line,
      predictedMean: value.predictedMean as number | null,
      recommendation,
      draft,
    };
  }

  return null;
}

export function parseBetLegsV2(value: unknown): BetLeg[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const parsed = parseBetLegV2(item);
    return parsed ? [parsed] : [];
  });
}

function lineToken(line: number | null | undefined): string {
  return line == null ? "none" : String(line);
}

function parseGameMarket(value: string): GameMarket {
  const market = parseGameMarketOrNull(value);
  if (!market) throw new Error(`Unsupported game market: ${value}`);
  return market;
}

function parseGameMarketOrNull(value: unknown): GameMarket | null {
  return typeof value === "string" && GAME_MARKETS.has(value as GameMarket)
    ? (value as GameMarket)
    : null;
}

function parseGameSide(value: string, market: GameMarket): GameSide {
  if (!isGameSideForMarket(value, market)) {
    throw new Error(`Unsupported side ${value} for ${market}`);
  }
  return value;
}

function isGameSideForMarket(
  value: unknown,
  market: GameMarket,
): value is GameSide {
  if (market === "total") return value === "over" || value === "under";
  return value === "home" || value === "away";
}

function isPropSide(value: unknown): value is PropSide {
  return typeof value === "string" && PROP_SIDES.has(value as PropSide);
}

function parseAmericanOdds(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    Number.isInteger(value) &&
    value !== 0
    ? value
    : null;
}

function probabilityOrNull(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value > 0 &&
    value < 1
    ? value
    : null;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonnegativeOrNull(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0
    ? value
    : null;
}

function isNullableFinite(value: unknown): boolean {
  return value === null || finiteOrNull(value) !== null;
}

function isNullableNonnegative(value: unknown): boolean {
  return value === null || nonnegativeOrNull(value) !== null;
}

function isNullableProbability(value: unknown): boolean {
  return value === null || probabilityOrNull(value) !== null;
}

function isNullableAmericanOdds(value: unknown): boolean {
  return value === null || parseAmericanOdds(value) !== null;
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isNonemptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isSource(value: unknown): value is BetLegSource {
  return typeof value === "string" && SOURCES.has(value as BetLegSource);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseRecommendation(value: unknown): RecommendationSnapshot | null {
  if (!isRecord(value) || !isNonemptyString(value.modelKey)) return null;
  if (
    !isNullableAmericanOdds(value.referenceAmericanOdds) ||
    !isNullableProbability(value.referenceModelProbability) ||
    !isNullableFinite(value.referenceModelValue) ||
    !isNullableFinite(value.referenceMarketValue) ||
    !isNullableFinite(value.referenceExpectedValue) ||
    !isNullableString(value.referenceEdgeStrength) ||
    !isNullableNonnegative(value.referenceKellyFraction) ||
    !isNullableNonnegative(value.referenceKellyStake) ||
    !isNullableNonnegative(value.referenceBankroll) ||
    !isNullableNonnegative(value.referenceKellyMultiplier)
  ) {
    return null;
  }

  return value as RecommendationSnapshot;
}

function parseDraft(value: unknown): BetDraftInputs | null {
  if (!isRecord(value)) return null;
  if (
    !isNullableAmericanOdds(value.currentAmericanOdds) ||
    !isNullableNonnegative(value.proposedStake) ||
    !isNullableString(value.sportsbook) ||
    !isNullableString(value.note)
  ) {
    return null;
  }
  return value as BetDraftInputs;
}
