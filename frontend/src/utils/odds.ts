/**
 * Odds format helpers ported from the prototype's app.jsx.
 * All pure — no state coupling.
 */

/** Convert American odds (-110, +150) to implied probability [0, 1]. */
export function impliedProb(americanOdds: number): number {
  if (americanOdds > 0) {
    return 100 / (americanOdds + 100);
  }
  return -americanOdds / (-americanOdds + 100);
}

/** Convert American odds to decimal odds. */
export function americanToDecimal(americanOdds: number): number {
  if (americanOdds > 0) {
    return 1 + americanOdds / 100;
  }
  return 1 + 100 / -americanOdds;
}

/** Convert decimal odds to American odds. Rounds to the nearest integer. */
export function decimalToAmerican(decimalOdds: number): number {
  if (decimalOdds >= 2) {
    return Math.round((decimalOdds - 1) * 100);
  }
  return Math.round(-100 / (decimalOdds - 1));
}

/**
 * Format American odds for display given a format preference.
 * Returns a signed string like "+150" or "-110" for American,
 * or a decimal like "2.50" for decimal.
 */
export function formatOdds(
  americanOdds: number,
  format: "american" | "decimal",
): string {
  if (format === "decimal") {
    return americanToDecimal(americanOdds).toFixed(2);
  }
  const sign = americanOdds >= 0 ? "+" : "";
  return `${sign}${americanOdds}`;
}

/**
 * Kelly fraction for a bet.
 *
 * @param p - Win probability (0 to 1).
 * @param b - Decimal odds minus 1 (e.g. American +150 → decimal 2.50 → b = 1.50).
 * @returns Kelly fraction — the fraction of bankroll to bet. Zero if edge is negative.
 */
export function kelly(p: number, b: number): number {
  if (b <= 0) return 0;
  const q = 1 - p;
  const f = (b * p - q) / b;
  return Math.max(0, f);
}

/**
 * Convert win probability (0-1) to American odds.
 *
 * Examples:
 *   probToAmerican(0.65) → -186 (favored)
 *   probToAmerican(0.35) → +186 (underdog)
 *   probToAmerican(0.50) → -100
 *
 * Returns 0 for invalid inputs (prob ≤ 0 or prob ≥ 1) as a defensive fallback.
 */
export function probToAmerican(prob: number): number {
  if (prob <= 0 || prob >= 1) return 0;
  if (prob >= 0.5) return Math.round((-100 * prob) / (1 - prob));
  return Math.round((100 * (1 - prob)) / prob);
}

/** Sportsbook names used across line-shopping views. */
export const BOOKS = [
  "DraftKings",
  "FanDuel",
  "BetMGM",
  "Caesars",
  "PointsBet",
  "BetRivers",
] as const;

export type Book = (typeof BOOKS)[number];
