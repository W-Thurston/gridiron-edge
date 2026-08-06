import type { components } from "../api/schema";

type EdgeRow = components["schemas"]["EdgeRow"];

export type SportsbookMode = "all" | "selected";

export type SportsbookPreference = {
  sportsbookMode: SportsbookMode;
  selectedSportsbooks: string[];
};

export type EdgeOfferGroup = {
  id: string;
  gameId: string;
  marketType: string;
  side: string;
  best: EdgeRow;
  alternatives: EdgeRow[];
  offers: EdgeRow[];
};

const DISPLAY_NAMES: Record<string, string> = {
  betmgm: "BetMGM",
  betonlineag: "BetOnline.ag",
  betrivers: "BetRivers",
  betus: "BetUS",
  bovada: "Bovada",
  draftkings: "DraftKings",
  fanduel: "FanDuel",
  lowvig: "LowVig",
  mybookieag: "MyBookie.ag",
};

export function normalizeSportsbookKey(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

export function normalizeSelectedSportsbooks(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return Array.from(
    new Set(value.map(normalizeSportsbookKey).filter((item): item is string => item !== null)),
  ).sort();
}

export function sportsbookDisplayName(key: string): string {
  return DISPLAY_NAMES[key] ?? key;
}

export function availableSportsbooks(edges: EdgeRow[]): string[] {
  return normalizeSelectedSportsbooks(edges.map((edge) => edge.sportsbook));
}

export function filterEdgesBySportsbook(
  edges: EdgeRow[],
  preference: SportsbookPreference,
): EdgeRow[] {
  if (preference.sportsbookMode === "all") return edges;
  const selected = new Set(normalizeSelectedSportsbooks(preference.selectedSportsbooks));
  return edges.filter((edge) => {
    const sportsbook = normalizeSportsbookKey(edge.sportsbook);
    return sportsbook === null || selected.has(sportsbook);
  });
}

export function edgeOfferKey(edge: EdgeRow): string {
  return [
    edge.provider_event_id ?? "no-event",
    normalizeSportsbookKey(edge.sportsbook) ?? "consensus",
    edge.game_id,
    edge.market_type,
    edge.side,
    edge.market_value ?? "no-line",
  ].join(":");
}

function compareEdgeOffers(left: EdgeRow, right: EdgeRow): number {
  const evOrder = right.ev - left.ev;
  if (evOrder !== 0) return evOrder;

  const sportsbookOrder = (
    normalizeSportsbookKey(left.sportsbook) ?? "consensus"
  ).localeCompare(normalizeSportsbookKey(right.sportsbook) ?? "consensus");
  if (sportsbookOrder !== 0) return sportsbookOrder;

  const priceOrder = right.american_odds - left.american_odds;
  if (priceOrder !== 0) return priceOrder;

  return (left.provider_event_id ?? "").localeCompare(
    right.provider_event_id ?? "",
  );
}

export function rankEdgeOffers(edges: EdgeRow[]): EdgeRow[] {
  return [...edges].sort(compareEdgeOffers);
}

export function selectBestEdge(edges: EdgeRow[]): EdgeRow | undefined {
  return rankEdgeOffers(edges)[0];
}

export function selectBestEdgeByMarket(
  edges: EdgeRow[],
  marketType: string,
): EdgeRow | undefined {
  return selectBestEdge(edges.filter((edge) => edge.market_type === marketType));
}

export function selectBestEdgePerGame(edges: EdgeRow[]): EdgeRow[] {
  const bestByGame = new Map<string, EdgeRow>();
  for (const edge of rankEdgeOffers(edges)) {
    if (!bestByGame.has(edge.game_id)) bestByGame.set(edge.game_id, edge);
  }
  return rankEdgeOffers(Array.from(bestByGame.values()));
}

export function edgeOfferGroupKey(edge: EdgeRow): string {
  return [edge.game_id, edge.market_type, edge.side].join(":");
}

export function groupEdgeOffers(edges: EdgeRow[]): EdgeOfferGroup[] {
  const grouped = new Map<string, EdgeRow[]>();

  for (const edge of edges) {
    const id = edgeOfferGroupKey(edge);
    const offers = grouped.get(id);
    if (offers) offers.push(edge);
    else grouped.set(id, [edge]);
  }

  const groups: EdgeOfferGroup[] = [];
  for (const [id, offers] of grouped) {
    const sportsbookOffers = offers.filter(
      (edge) => normalizeSportsbookKey(edge.sportsbook) !== null,
    );
    const actionable = sportsbookOffers.length > 0 ? sportsbookOffers : offers;
    const ranked = rankEdgeOffers(actionable);
    const best = ranked[0];
    if (!best) continue;

    groups.push({
      id,
      gameId: best.game_id,
      marketType: best.market_type,
      side: best.side,
      best,
      alternatives: ranked.slice(1),
      offers: ranked,
    });
  }

  return groups.sort((left, right) =>
    compareEdgeOffers(left.best, right.best),
  );
}
