import { useMemo, useState } from "react";
import type { components } from "../api/schema";
import { useLines } from "../api/hooks";
import { ErrorCard } from "../components/error/ErrorCard";
import { ExplainTooltip, type ExplainTooltipSection } from "../components/primitives/ExplainTooltip";
import { Pill } from "../components/primitives/Pill";
import { TeamMark } from "../components/primitives/TeamMark";
import { useAppState } from "../context/AppStateContext";
import {
  normalizeSportsbookKey,
  sportsbookDisplayName,
} from "../utils/sportsbookPreferences";

type Market = "spread" | "total" | "moneyline";
type LineOffer = components["schemas"]["LineOffer"];
type LineShoppingGame = components["schemas"]["LineShoppingGame"];
type LineOutcomeGuidance = components["schemas"]["LineOutcomeGuidance"];
type Side = LineOffer["side"];

const MARKET_OPTIONS: { value: Market; label: string }[] = [
  { value: "spread", label: "Spread" },
  { value: "total", label: "Total" },
  { value: "moneyline", label: "Moneyline" },
];

const SIDE_ORDER: Record<Market, Side[]> = {
  spread: ["away", "home"],
  total: ["over", "under"],
  moneyline: ["away", "home"],
};

export function LineShopping() {
  const [market, setMarket] = useState<Market>("spread");
  const { state, setState } = useAppState();
  const { data, isLoading, error, refetch } = useLines({ market });

  const selected = useMemo(
    () => new Set(state.selectedSportsbooks.map(normalizeSportsbookKey)),
    [state.selectedSportsbooks],
  );
  const sportsbooks = (data?.sportsbooks ?? []).filter((sportsbook) =>
    state.sportsbookMode === "all"
      ? true
      : selected.has(normalizeSportsbookKey(sportsbook)),
  );
  const games = (data?.items ?? []).map((game) => ({
    ...game,
    offers: (game.offers ?? []).filter((offer) =>
      sportsbooks.includes(offer.sportsbook),
    ),
  }));
  const visibleOfferCount = games.reduce(
    (total, game) => total + game.offers.length,
    0,
  );
  const blocked = blockedItems(data?._meta?.field_status?.items);
  const updatedAt = latestTimestamp(data?.market_fetched_at ?? []);

  return (
    <div className="line-shopping-screen">
      <header className="line-shopping-header">
        <div>
          <div id="line-shopping-heading" className="upper dim" style={{ fontSize: 10 }}>
            Line Shopping
          </div>
          <div className="mono dim2" style={{ fontSize: 11, marginTop: 5 }}>
            {data ? `Week ${data.week} · ${sportsbooks.length} available sportsbooks` : "Current multi-book markets"}
          </div>
        </div>
        <div className="line-shopping-control-stack">
          <div className="line-shopping-controls" aria-label="Market type">
            {MARKET_OPTIONS.map((option) => (
              <Pill
                key={option.value}
                active={market === option.value}
                onClick={() => setMarket(option.value)}
              >
                {option.label}
              </Pill>
            ))}
          </div>
          <button
            type="button"
            className="line-shopping-highlight-toggle"
            aria-pressed={state.lineShoppingHighlights}
            onClick={() => setState({
              lineShoppingHighlights: !state.lineShoppingHighlights,
            })}
          >
            Highlight model guidance
            <span aria-hidden="true">
              {state.lineShoppingHighlights ? "On" : "Off"}
            </span>
          </button>
        </div>
      </header>

      <div className="line-shopping-legend mono dim" aria-label="Comparison legend">
        <div className="line-shopping-legend-keys">
        {state.lineShoppingHighlights && (
          <>
            <span><i className="line-shopping-key line-shopping-key--approved" />Model approved</span>
            <span><i className="line-shopping-key line-shopping-key--line" />Best available line</span>
            <span><i className="line-shopping-key line-shopping-key--preferred" />Preferred model-approved offer</span>
            <span>
              <strong className="line-shopping-best-price">Orange price</strong> is best at this exact line
            </span>
          </>
        )}
        </div>
        <span className="line-shopping-snapshot">
          {updatedAt ? `Snapshot ${formatTimestamp(updatedAt)}` : "Snapshot time unavailable"}
        </span>
      </div>

      {isLoading && (
        <div className="hm-card line-shopping-status" role="status">
          Loading current sportsbook offers…
        </div>
      )}

      {error && (
        <ErrorCard
          error={error}
          onRetry={() => void refetch()}
          title="Unable to load Line Shopping"
        />
      )}

      {!isLoading && !error && blocked && (
        <div className="hm-card line-shopping-status" role="status">
          <div style={{ color: "var(--warn)", fontWeight: 600 }}>
            Current sportsbook offers are unavailable.
          </div>
          <div className="mono dim2" style={{ marginTop: 6 }}>
            Reason: {blocked.blocker}
          </div>
        </div>
      )}

      {!isLoading && !error && !blocked && data && (data.items ?? []).length === 0 && (
        <div className="hm-card line-shopping-status" role="status">
          No current {marketLabel(market).toLowerCase()} offers for this season and week.
        </div>
      )}

      {!isLoading && !error && !blocked && data && (data.items ?? []).length > 0 && visibleOfferCount === 0 && (
        <div className="hm-card line-shopping-status" role="status">
          No offers are available from your selected sportsbooks.
        </div>
      )}

      {!isLoading && !error && !blocked && visibleOfferCount > 0 && (
        <div
          className="market-table-scroll line-shopping-scroll"
          role="region"
          aria-labelledby="line-shopping-heading"
          tabIndex={0}
        >
          <table
            className="line-shopping-table mono tnum"
            style={{ minWidth: `${Math.max(760, 330 + sportsbooks.length * 140)}px` }}
          >
            <caption className="visually-hidden">
              Current {marketLabel(market).toLowerCase()} offers by matchup, outcome, and sportsbook
            </caption>
            <thead>
              <tr>
                <th scope="col">Matchup</th>
                <th scope="col">Outcome</th>
                {sportsbooks.map((sportsbook) => (
                  <th key={sportsbook} scope="col">
                    {sportsbookDisplayName(sportsbook)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {games.map((game) => (
                <GameRows
                  key={game.game_id}
                  game={game}
                  market={market}
                  sportsbooks={sportsbooks}
                  highlights={state.lineShoppingHighlights}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function GameRows({
  game,
  market,
  sportsbooks,
  highlights,
}: {
  game: LineShoppingGame;
  market: Market;
  sportsbooks: string[];
  highlights: boolean;
}) {
  return SIDE_ORDER[market].map((side, index) => (
    <tr key={`${game.game_id}:${side}`}>
      {index === 0 && (
        <th className="line-shopping-matchup" scope="rowgroup" rowSpan={2}>
          <span><TeamMark abbr={game.away_team} size={18} /> {game.away_team}</span>
          <span><TeamMark abbr={game.home_team} size={18} /> {game.home_team}</span>
          <span className="dim2" style={{ fontSize: 10 }}>
            {formatKickoff(game.commence_time, game.game_date)}
          </span>
        </th>
      )}
      <th className="line-shopping-outcome" scope="row">
        <span>{outcomeLabel(game, side)}</span>
        <OutcomeGuidance
          game={game}
          guidance={(game.guidance ?? []).find((item) => item.side === side)}
          market={market}
          side={side}
        />
      </th>
      {sportsbooks.map((sportsbook) => {
        const offer = game.offers?.find(
          (candidate) => candidate.sportsbook === sportsbook && candidate.side === side,
        );
        return (
          <td key={sportsbook}>
            {offer ? (
              <OfferCell
                game={game}
                offer={offer}
                highlights={highlights}
              />
            ) : (
              <span className="dim2">Unavailable</span>
            )}
          </td>
        );
      })}
    </tr>
  ));
}

function OutcomeGuidance({
  game,
  guidance,
  market,
  side,
}: {
  game: LineShoppingGame;
  guidance: LineOutcomeGuidance | undefined;
  market: Market;
  side: Side;
}) {
  if (!guidance || guidance.model_status === "model_unavailable") {
    return <span className="line-shopping-guidance dim2">Model guidance unavailable</span>;
  }
  if (guidance.model_status === "uncertainty_unavailable") {
    return <span className="line-shopping-guidance dim2">Model uncertainty unavailable</span>;
  }

  const teamName = side === "away" ? game.away_team : game.home_team;
  const sections = outcomeExplanationSections(game, guidance, market, side);
  let visible: React.ReactNode;
  if (market === "moneyline" && guidance.fair_american_odds != null) {
    visible = (
      <>
        <span>Model win chance {formatProbability(guidance.model_value)}</span>
        <span>Fair price {formatAmericanOdds(guidance.fair_american_odds)}</span>
      </>
    );
  } else if (
    guidance.model_value != null
    && guidance.playable_line != null
    && guidance.reference_odds != null
  ) {
    const modelValue = formatGuidanceLine(guidance.model_value);
    const boundary = formatGuidanceLine(guidance.playable_line);
    const referencePrice = formatAmericanOdds(guidance.reference_odds);
    visible = market === "total" ? (
      <>
        <span>Model total {modelValue}</span>
        <span>{side === "over" ? "Over" : "Under"} +EV {side === "over" ? "below" : "above"} {boundary} at {referencePrice}</span>
      </>
    ) : (
      <>
        <span>Model {teamName} {modelValue}</span>
        <span>Playable {teamName} {boundary} or more at {referencePrice}</span>
      </>
    );
  } else {
    return <span className="line-shopping-guidance dim2">Model guidance unavailable</span>;
  }

  return (
    <div className="line-shopping-guidance-row">
      <span className="line-shopping-guidance dim2">{visible}</span>
      <ExplainTooltip
        accessibleLabel={`Explain ${teamName} ${market} model guidance`}
        title={`${teamName} model guidance`}
        sections={sections}
        className="line-shopping-info-button"
      >
        <span aria-hidden="true">i</span>
      </ExplainTooltip>
    </div>
  );
}

function OfferCell({
  game,
  offer,
  highlights,
}: {
  game: LineShoppingGame;
  offer: LineOffer;
  highlights: boolean;
}) {
  const classNames = ["line-shopping-offer"];
  if (highlights && offer.is_model_approved === true) {
    classNames.push("line-shopping-offer--approved");
  }
  if (highlights && offer.is_best_line) {
    classNames.push("line-shopping-offer--best-line");
  }
  if (highlights && offer.is_best_model_approved_offer) {
    classNames.push("line-shopping-offer--preferred");
  }
  const title = offerTitle(game, offer);
  return (
    <ExplainTooltip
      accessibleLabel={`Explain ${title}`}
      title={title}
      sections={offerExplanationSections(game, offer)}
      className={classNames.join(" ")}
    >
      <span className="line-shopping-offer-values">
        {offer.market === "moneyline" ? null : <span>{formatLine(offer)}</span>}
        <strong className={highlights && offer.is_best_price ? "line-shopping-best-price" : undefined}>
          {formatAmericanOdds(offer.american_odds)}
        </strong>
      </span>
    </ExplainTooltip>
  );
}

function outcomeExplanationSections(
  game: LineShoppingGame,
  guidance: LineOutcomeGuidance,
  market: Market,
  side: Side,
): ExplainTooltipSection[] {
  const team = side === "away" ? game.away_team : game.home_team;
  const opponent = side === "away" ? game.home_team : game.away_team;
  if (market === "moneyline") {
    return [{
      label: "Model",
      text: `The model estimates a ${formatProbability(guidance.model_value)} chance that ${team} wins against ${opponent}. That probability corresponds to fair odds of ${formatAmericanOdds(guidance.fair_american_odds ?? 0)}.`,
    }];
  }
  if (market === "total") {
    const direction = side === "over" ? "below" : "above";
    const reason = side === "over"
      ? "fewer points are needed for the Over to win"
      : "more points can be scored before the Under loses";
    return [{
      label: "Model",
      text: `The model projects ${formatGuidanceLine(guidance.model_value ?? 0)} combined points. At ${formatAmericanOdds(guidance.reference_odds ?? -110)}, the ${side === "over" ? "Over" : "Under"} has positive expected value ${direction} ${formatGuidanceLine(guidance.playable_line ?? 0)} because ${reason}.`,
    }];
  }
  const modelLine = formatGuidanceLine(guidance.model_value ?? 0);
  const playable = formatGuidanceLine(guidance.playable_line ?? 0);
  return [{
    label: "Model",
    text: `The model projects ${team} at ${modelLine}. At ${formatAmericanOdds(guidance.reference_odds ?? -110)}, ${team} ${playable} or any numerically larger spread has positive expected value. For a favorite, -3.5 is larger and more favorable than -4.5. For an underdog, +4.5 is larger and more favorable than +3.5.`,
  }];
}

function offerExplanationSections(
  game: LineShoppingGame,
  offer: LineOffer,
): ExplainTooltipSection[] {
  const sections: ExplainTooltipSection[] = [
    { label: "Bet outcome", text: betOutcomeExplanation(game, offer) },
    { label: "Price", text: priceExplanation(offer.american_odds) },
    { label: "Model", text: offerModelExplanation(offer) },
  ];
  const marketFacts = [
    offer.is_best_line ? "This is the best available line." : null,
    offer.is_best_price ? "This is the best price available at this exact line." : null,
    offer.is_best_model_approved_offer ? "This is a preferred model-approved offer." : null,
  ].filter((value): value is string => value !== null);
  if (marketFacts.length > 0) {
    sections.push({ label: "Market", text: marketFacts.join(" ") });
  }
  return sections;
}

function betOutcomeExplanation(game: LineShoppingGame, offer: LineOffer): string {
  const team = offer.side === "away" ? game.away_team : game.home_team;
  if (offer.market === "moneyline") {
    return `This wager wins if ${team} wins the game. The margin of victory does not matter.`;
  }
  const line = offer.line ?? 0;
  const absolute = Math.abs(line);
  const whole = Number.isInteger(absolute);
  if (offer.market === "total") {
    const threshold = whole
      ? absolute + (offer.side === "over" ? 1 : -1)
      : offer.side === "over" ? Math.ceil(absolute) : Math.floor(absolute);
    const result = `${offer.side === "over" ? "Over" : "Under"} ${absolute.toFixed(1)} wins if the teams combine for ${threshold} points ${offer.side === "over" ? "or more" : "or fewer"}.`;
    return whole
      ? `${result} If the teams combine for exactly ${absolute.toFixed(0)} points, the wager pushes and the stake is returned.`
      : result;
  }
  if (line > 0) {
    const allowedLoss = whole ? absolute - 1 : Math.floor(absolute);
    const result = `${team} ${formatGuidanceLine(line)} wins if ${team} wins or loses by ${allowedLoss} points or fewer.`;
    return whole
      ? `${result} If ${team} loses by exactly ${absolute.toFixed(0)} points, the wager pushes and the stake is returned.`
      : result;
  }
  const requiredMargin = whole ? absolute + 1 : Math.ceil(absolute);
  const result = `${team} ${formatGuidanceLine(line)} wins if ${team} wins by ${requiredMargin} points or more.`;
  return whole
    ? `${result} If ${team} wins by exactly ${absolute.toFixed(0)} points, the wager pushes and the stake is returned.`
    : result;
}

function priceExplanation(odds: number): string {
  if (odds > 0) {
    return `At ${formatAmericanOdds(odds)}, a $100 stake would produce $${odds} in profit and a $${odds + 100} total return, including the original stake.`;
  }
  const stake = Math.abs(odds);
  return `At ${odds}, a $${stake} stake would produce $100 in profit and a $${stake + 100} total return, including the original stake.`;
}

function offerModelExplanation(offer: LineOffer): string {
  if (offer.model_status === "model_unavailable") {
    return "Model guidance is unavailable for this offer.";
  }
  if (offer.model_status === "uncertainty_unavailable") {
    return "Model uncertainty is unavailable for this offer.";
  }
  const decision = offer.is_model_approved
    ? "The model approves this exact line and price."
    : "The model does not approve this exact line and price.";
  return offer.expected_value == null
    ? decision
    : `${decision} Expected value: ${formatPercent(offer.expected_value)}.`;
}

function offerTitle(game: LineShoppingGame, offer: LineOffer): string {
  const team = offer.side === "away" ? game.away_team : game.home_team;
  const market = offer.market === "moneyline"
    ? "moneyline"
    : formatLine(offer);
  return `${team} ${market} at ${formatAmericanOdds(offer.american_odds)}`;
}

function blockedItems(status: unknown): { blocker: string; roadmap: string } | null {
  if (!status || typeof status !== "object") return null;
  const value = status as Record<string, unknown>;
  if (value.status !== "blocked" || typeof value.blocker !== "string") return null;
  return {
    blocker: value.blocker,
    roadmap: typeof value.roadmap === "string" ? value.roadmap : "",
  };
}

function latestTimestamp(values: string[]): string | null {
  return values.length === 0 ? null : [...values].sort().at(-1) ?? null;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function marketLabel(market: Market): string {
  return market === "moneyline" ? "Moneyline" : market[0].toUpperCase() + market.slice(1);
}

function outcomeLabel(game: LineShoppingGame, side: Side): string {
  if (side === "away") return game.away_team;
  if (side === "home") return game.home_team;
  return side === "over" ? "Over" : "Under";
}

function formatAmericanOdds(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function formatKickoff(
  commenceTime: string | null | undefined,
  fallbackDate: string,
): string {
  if (!commenceTime) return fallbackDate;
  const date = new Date(commenceTime);
  if (Number.isNaN(date.getTime())) return fallbackDate;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((item) => item.type === type)?.value ?? "";
  return [
    part("weekday").toUpperCase(),
    `${part("month").toUpperCase()} ${part("day")}`,
    `${part("hour")}:${part("minute")} ${part("dayPeriod").toUpperCase()} ET`,
  ].join(" · ");
}

function formatProbability(value: number | null | undefined): string {
  return value == null ? "unavailable" : `${(value * 100).toFixed(1)}%`;
}

function formatPercent(value: number): string {
  const rendered = `${(value * 100).toFixed(1)}%`;
  return value > 0 ? `+${rendered}` : rendered;
}

function formatGuidanceLine(value: number): string {
  const rendered = value.toFixed(1);
  return value > 0 ? `+${rendered}` : rendered;
}

function formatLine(offer: LineOffer): string {
  if (offer.line == null) return "";
  if (offer.market === "total") return `${offer.side === "over" ? "O" : "U"} ${offer.line.toFixed(1)}`;
  return `${offer.line > 0 ? "+" : ""}${offer.line.toFixed(1)}`;
}
