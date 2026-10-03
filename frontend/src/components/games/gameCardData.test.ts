import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import {
  buildGamesCardData,
  selectBestVisibleOffer,
} from "./gameCardData";

type GameSummary = components["schemas"]["GameSummary"];
type LineOffer = components["schemas"]["LineOffer"];
type LineShoppingGame = components["schemas"]["LineShoppingGame"];

function game(overrides: Partial<GameSummary> = {}): GameSummary {
  return {
    game_id: "2026_01_KC_LAC",
    game_date: "2026-09-05",
    season: "2026-2027",
    week: 1,
    away_team: "KC",
    home_team: "LAC",
    win: { status: "available", away_win_prob: 0.42, home_win_prob: 0.58 },
    spread: { status: "available", model_spread: -2.5 },
    total: { status: "available", model_total: 47.5 },
    projected_score: { status: "available", away: 22.5, home: 25 },
    starting_qb: { status: "not_recorded" },
    ...overrides,
  };
}

function offer(overrides: Partial<LineOffer> = {}): LineOffer {
  return {
    american_odds: -110,
    is_best_line: false,
    is_best_model_approved_offer: false,
    is_best_price: false,
    is_live: false,
    is_model_approved: false,
    is_model_recommended_offer: false,
    is_model_recommended_side: false,
    line: null,
    market: "moneyline",
    market_fetched_at: "2026-09-05T12:00:00Z",
    model_probability: 0.55,
    model_status: "available",
    provider: "the_odds_api",
    provider_event_id: "event-1",
    side: "away",
    sportsbook: "draftkings",
    ...overrides,
  };
}

function lineGame(
  offers: LineOffer[],
  overrides: Partial<LineShoppingGame> = {},
): LineShoppingGame {
  return {
    game_id: "2026_01_KC_LAC",
    season: "2026-2027",
    week: 1,
    game_date: "2026-09-05",
    away_team: "KC",
    home_team: "LAC",
    offers,
    guidance: [],
    ...overrides,
  };
}

describe("buildGamesCardData", () => {
  it("keeps every scheduled game when Line Shopping evidence is absent", () => {
    const cards = buildGamesCardData({
      games: [game(), game({ game_id: "2026_01_BUF_MIA" })],
      lineGames: [],
      visibleSportsbooks: ["draftkings"],
      market: "moneyline",
    });

    expect(cards).toHaveLength(2);
    expect(cards[0].lineGame).toBeNull();
    expect(cards[0].offers.every((item) => item.offer === null)).toBe(true);
  });

  it("joins only by game_id and filters hidden sportsbooks", () => {
    const hidden = offer({ sportsbook: "fanduel", american_odds: 130 });
    const visible = offer({ sportsbook: "draftkings", american_odds: 110 });
    const cards = buildGamesCardData({
      games: [game()],
      lineGames: [lineGame([hidden, visible])],
      visibleSportsbooks: ["draftkings"],
      market: "moneyline",
    });

    expect(cards[0].offers[0].offer).toBe(visible);
  });

  it("uses the shared visible recommendation result as the primary offer", () => {
    const neutral = offer({
      is_best_price: true,
      american_odds: 120,
    });
    const recommended = offer({
      sportsbook: "fanduel",
      american_odds: 110,
      is_model_recommended_side: true,
      model_probability: 0.6,
    });
    const cards = buildGamesCardData({
      games: [game()],
      lineGames: [lineGame([neutral, recommended])],
      visibleSportsbooks: ["draftkings", "fanduel"],
      market: "moneyline",
    });

    expect(cards[0].offers[0]).toMatchObject({
      offer: recommended,
      recommended: true,
    });
  });

  it("does not infer recommendation from EV, best-line, or best-price flags", () => {
    const analytical = offer({
      expected_value: 0.12,
      is_model_approved: true,
      is_best_line: true,
      is_best_price: true,
    });
    const cards = buildGamesCardData({
      games: [game()],
      lineGames: [lineGame([analytical])],
      visibleSportsbooks: ["draftkings"],
      market: "moneyline",
    });

    expect(cards[0].offers[0]).toMatchObject({
      offer: analytical,
      recommended: false,
    });
  });

  it("selects offers only from the active market", () => {
    const moneyline = offer();
    const spread = offer({
      market: "spread",
      side: "home",
      line: -3.5,
      is_best_line: true,
      is_best_price: true,
    });
    const cards = buildGamesCardData({
      games: [game()],
      lineGames: [lineGame([moneyline, spread])],
      visibleSportsbooks: ["draftkings"],
      market: "spread",
    });

    expect(cards[0].offers.find((item) => item.side === "home")?.offer).toBe(spread);
    expect(cards[0].offers.some((item) => item.offer === moneyline)).toBe(false);
  });
});

describe("selectBestVisibleOffer", () => {
  it("prefers persisted best-line, then best-price, then American odds", () => {
    const worseLine = offer({ market: "spread", side: "away", line: 2.5, american_odds: 120 });
    const bestLine = offer({ market: "spread", side: "away", line: 3.5, is_best_line: true, american_odds: -115 });
    const bestLineAndPrice = offer({ market: "spread", side: "away", line: 3.5, is_best_line: true, is_best_price: true, american_odds: -110 });

    expect(selectBestVisibleOffer([worseLine, bestLine, bestLineAndPrice])).toBe(bestLineAndPrice);
  });

  it("does not mutate the source offers", () => {
    const first = offer({ sportsbook: "fanduel", american_odds: -105 });
    const second = offer({ sportsbook: "draftkings", american_odds: -110 });
    const offers = [first, second];
    const original = [...offers];

    selectBestVisibleOffer(offers);
    expect(offers).toEqual(original);
  });
});
