import { describe, expect, it } from "vitest";
import type { components } from "../api/schema";
import {
  availableSportsbooks,
  edgeOfferKey,
  filterEdgesBySportsbook,
  normalizeSelectedSportsbooks,
  selectBestEdge,
  selectBestEdgePerGame,
  sportsbookDisplayName,
} from "./sportsbookPreferences";

type EdgeRow = components["schemas"]["EdgeRow"];

function edge(sportsbook: string | null): EdgeRow {
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
});
