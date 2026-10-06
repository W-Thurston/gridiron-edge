import type { components } from "../../api/schema";

type WagerSummary = components["schemas"]["RecapWagerSummary"];

const MARKET_LABELS: Record<string, string> = {
  moneyline: "Moneyline",
  spread: "Spread",
  total: "Total",
  player_prop: "Player prop",
  parlay: "Parlay",
  same_game_parlay: "Same-game parlay",
  special: "Special",
};

export function marketLabel(value: string): string {
  return MARKET_LABELS[value] ?? value.replaceAll("_", " ");
}

export function record(summary: WagerSummary): string {
  return `${summary.wins}-${summary.losses}-${summary.pushes}`;
}

export function money(value: number): string {
  return `$${value.toFixed(2)}`;
}

/** Signed dollars with a true minus sign, so the sign never relies on color. */
export function signedMoney(value: number): string {
  return `${value < 0 ? "−" : "+"}$${Math.abs(value).toFixed(2)}`;
}

export function roi(value: number | null | undefined): string {
  return value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}
