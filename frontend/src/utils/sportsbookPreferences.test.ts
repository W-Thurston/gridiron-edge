import { describe, expect, it } from "vitest";
import type { components } from "../api/schema";
import {
  availableSportsbooks,
  edgeOfferKey,
  filterEdgesBySportsbook,
  groupEdgeOffers,
  normalizeSelectedSportsbooks,
  selectBestEdge,
  selectBestEdgePerGame,
  sportsbookDisplayName,
} from "./sportsbookPreferences";

type EdgeRow = components["schemas"]["EdgeRow"];

function edge(
  sportsbook: string | null,
  overrides: Partial<EdgeRow> = {},
): EdgeRow {
  return {
    provider: sportsbook ? "the_odds_api" : "nflverse",
    sportsbook,
    game_id: "2026_01_KC_LAC",
    away_team: "KC",
    home_team: "LAC",
    model_key: "win_prob_elo",
    market_type: "moneyline",
    side: "home",
    american_odds: -110,
    ev: 0.08,
    edge_strength: "strong",
    ...overrides,
  };
}

describe("sportsbook preferences", () => {
  it("normalizes, deduplicates, and sorts selected keys", () => {
    expect(normalizeSelectedSportsbooks([" FanDuel ", "draftkings", "FANDUEL", null])).toEqual([
      "draftkings",
      "fanduel",
    ]);
  });

  it("collects only current sportsbook offers", () => {
    expect(availableSportsbooks([edge("fanduel"), edge(null), edge("draftkings")])).toEqual([
      "draftkings",
      "fanduel",
    ]);
  });

  it("keeps every offer in all mode", () => {
    const edges = [edge("draftkings"), edge("fanduel"), edge(null)];
    expect(filterEdgesBySportsbook(edges, {
      sportsbookMode: "all",
      selectedSportsbooks: [],
    })).toEqual(edges);
  });

  it("filters book offers while retaining truthful consensus rows", () => {
    const result = filterEdgesBySportsbook(
      [edge("draftkings"), edge("fanduel"), edge(null)],
      { sportsbookMode: "selected", selectedSportsbooks: ["fanduel"] },
    );
    expect(result.map((item) => item.sportsbook)).toEqual(["fanduel", null]);
  });

  it("renders friendly known labels and preserves unknown keys", () => {
    expect(sportsbookDisplayName("draftkings")).toBe("DraftKings");
    expect(sportsbookDisplayName("futurebook")).toBe("futurebook");
  });


  it("builds distinct stable keys for sportsbook offers", () => {
    expect(edgeOfferKey(edge("draftkings"))).not.toBe(
      edgeOfferKey(edge("fanduel")),
    );
    expect(edgeOfferKey(edge(null))).toContain("consensus");
  });


  it("selects the highest-EV offer deterministically", () => {
    const draftkings = { ...edge("draftkings"), ev: 0.08, american_odds: -150 };
    const fanduel = { ...edge("fanduel"), ev: 0.1, american_odds: -140 };
    expect(selectBestEdge([draftkings, fanduel])?.sportsbook).toBe("fanduel");
  });

  it("uses sportsbook then price then event identity to break EV ties", () => {
    const fanduel = {
      ...edge("fanduel"),
      ev: 0.1,
      american_odds: -140,
      provider_event_id: "event-b",
    };
    const draftkings = {
      ...edge("draftkings"),
      ev: 0.1,
      american_odds: -150,
      provider_event_id: "event-a",
    };
    expect(selectBestEdge([fanduel, draftkings])?.sportsbook).toBe("draftkings");
  });

  it("selects one best offer per game", () => {
    const firstGame = edge("draftkings");
    const betterFirstGame = { ...edge("fanduel"), ev: 0.1 };
    const secondGame = {
      ...edge("betmgm"),
      game_id: "2026_01_BUF_MIA",
      ev: 0.09,
    };
    const selected = selectBestEdgePerGame([
      firstGame,
      betterFirstGame,
      secondGame,
    ]);
    expect(selected).toHaveLength(2);
    expect(selected.map((item) => item.sportsbook)).toEqual(["fanduel", "betmgm"]);
  });


  it("groups sportsbook offers by game market and side", () => {
    const groups = groupEdgeOffers([
      edge("draftkings", { ev: 0.08, american_odds: -150 }),
      edge("fanduel", { ev: 0.1, american_odds: -140 }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({
      id: "2026_01_KC_LAC:moneyline:home",
      gameId: "2026_01_KC_LAC",
      marketType: "moneyline",
      side: "home",
    });
    expect(groups[0]?.best.sportsbook).toBe("fanduel");
    expect(groups[0]?.alternatives.map((offer) => offer.sportsbook)).toEqual([
      "draftkings",
    ]);
  });

  it("keeps different markets and sides in distinct wager families", () => {
    const groups = groupEdgeOffers([
      edge("draftkings"),
      edge("draftkings", { side: "away" }),
      edge("draftkings", { market_type: "spread", market_value: -3.5 }),
    ]);

    expect(groups.map((group) => group.id).sort()).toEqual([
      "2026_01_KC_LAC:moneyline:away",
      "2026_01_KC_LAC:moneyline:home",
      "2026_01_KC_LAC:spread:home",
    ]);
  });

  it("preserves each alternative line price and provenance", () => {
    const groups = groupEdgeOffers([
      edge("draftkings", {
        market_type: "spread",
        market_value: -3.5,
        american_odds: -110,
        ev: 0.08,
        provider_event_id: "event-dk",
      }),
      edge("fanduel", {
        market_type: "spread",
        market_value: -4,
        american_odds: 105,
        ev: 0.1,
        provider_event_id: "event-fd",
      }),
    ]);

    expect(groups[0]?.offers).toEqual([
      expect.objectContaining({
        sportsbook: "fanduel",
        market_value: -4,
        american_odds: 105,
        provider_event_id: "event-fd",
      }),
      expect.objectContaining({
        sportsbook: "draftkings",
        market_value: -3.5,
        american_odds: -110,
        provider_event_id: "event-dk",
      }),
    ]);
  });

  it("uses consensus only when no sportsbook offer exists", () => {
    const withBook = groupEdgeOffers([
      edge(null, { ev: 0.2 }),
      edge("draftkings", { ev: 0.08 }),
    ]);
    expect(withBook[0]?.offers.map((offer) => offer.sportsbook)).toEqual([
      "draftkings",
    ]);

    const consensusOnly = groupEdgeOffers([edge(null, { ev: 0.2 })]);
    expect(consensusOnly[0]?.best.sportsbook).toBeNull();
    expect(consensusOnly[0]?.alternatives).toEqual([]);
  });

  it("ranks wager families by their best offers without mutating inputs", () => {
    const lower = edge("draftkings", { ev: 0.08 });
    const higher = edge("fanduel", {
      game_id: "2026_01_BUF_MIA",
      ev: 0.12,
    });
    const input = [lower, higher];

    const groups = groupEdgeOffers(input);

    expect(groups.map((group) => group.gameId)).toEqual([
      "2026_01_BUF_MIA",
      "2026_01_KC_LAC",
    ]);
    expect(input).toEqual([lower, higher]);
  });

});
