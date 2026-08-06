import {
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import type {
  components,
} from "../../api/schema";
import {
  createGameBetLeg,
  createPropBetLeg,
  type BetLeg,
} from "../../utils/betLegs";
import type {
  BetSlipSizingResult,
} from "../../hooks/useBetSlipSizing";
import { TestWrapper } from "../../test/testWrapper";
import { SlipPanel } from "./SlipPanel";

type EdgeApiRow =
  components["schemas"]["EdgeRow"];

type PropApi =
  components["schemas"]["PropSummary"];

const ADDED_AT =
  "2026-07-29T22:00:00.000Z";

const sizing: BetSlipSizingResult = {
  preference: {
    version: 1,
    bankrollMode: "tracked",
    whatIfBankroll: null,
    kellyMultiplier: 0.25,
  },
  bankroll: 2500,
  bankrollSource: "tracked",
  trackedBankroll: 2500,
  bankrollMode: "tracked",
  whatIfBankroll: null,
  kellyMultiplier: 0.25,
  isTrackedBankrollLoading:
    false,
  trackedBankrollError: null,
  updateSizing: vi.fn(),
  setBankrollMode: vi.fn(),
  setWhatIfBankroll: vi.fn(),
  setKellyMultiplier: vi.fn(),
};

function edge({
  gameId,
  awayTeam,
  homeTeam,
  americanOdds,
  side,
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  americanOdds: number;
  side: "home" | "away";
}): EdgeApiRow {
  return {
    american_odds:
      americanOdds,
    away_team: awayTeam,
    cover_prob: null,
    edge_strength: "strong",
    ev: 0.08,
    game_id: gameId,
    home_team: homeTeam,
    kelly_frac: 0.08,
    kelly_stake: 20,
    market_type: "moneyline",
    market_value: 0.45,
    model_key:
      "random_forest_win_prob",
    model_value: 0.58,
    point_edge: null,
    side,
  };
}

function gameLeg({
  gameId,
  awayTeam,
  homeTeam,
  americanOdds,
  side,
  proposedStake,
}: {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  americanOdds: number;
  side: "home" | "away";
  proposedStake: number | null;
}): BetLeg {
  const leg = createGameBetLeg({
    edge: edge({
      gameId,
      awayTeam,
      homeTeam,
      americanOdds,
      side,
    }),
    source: "betslip-edges",
    addedAt: ADDED_AT,
    referenceBankroll: 2500,
    referenceKellyMultiplier:
      0.25,
  });

  return {
    ...leg,
    draft: {
      ...leg.draft,
      proposedStake,
    },
  };
}

function propLeg(): BetLeg {
  const prop: PropApi = {
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

  return createPropBetLeg({
    prop,
    side: "over",
    source:
      "dashboard-prop-edges",
    addedAt: ADDED_AT,
  });
}

function storeLegs(legs: BetLeg[]) {
  localStorage.setItem(
    "hm-betslip-v3",
    JSON.stringify(legs),
  );
}

function storeMode(
  mode: "single" | "parlay",
) {
  localStorage.setItem(
    "hm-betslip-mode-v3",
    mode,
  );
}

function renderPanel() {
  return render(
    <TestWrapper>
      <SlipPanel sizing={sizing} />
    </TestWrapper>,
  );
}

describe("SlipPanel summary", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it(
    "shows complete aggregate economics for singles",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: 110,
        }),
        gameLeg({
          gameId:
            "2026_01_BUF_MIA",
          awayTeam: "BUF",
          homeTeam: "MIA",
          americanOdds: 150,
          side: "home",
          proposedStake: 100,
        }),
      ]);

      storeMode("single");
      renderPanel();

      const summary =
        screen.getByLabelText(
          "Bet slip summary",
        );

      expect(summary).toHaveAttribute(
        "aria-live",
        "polite",
      );

      expect(summary).toHaveAttribute(
        "aria-atomic",
        "true",
      );

      expect(summary).toHaveTextContent(
        "Singles Summary",
      );

      expect(summary).toHaveTextContent(
        "$210.00",
      );

      expect(summary).toHaveTextContent(
        "$460.00",
      );

      expect(summary).toHaveTextContent(
        "+$250.00",
      );

      expect(summary).not.toHaveTextContent(
        "Summary incomplete",
      );
    },
  );

  it(
    "blocks the full singles summary when one leg is incomplete",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: 110,
        }),
        propLeg(),
      ]);

      storeMode("single");
      renderPanel();

      const summary =
        screen.getByLabelText(
          "Bet slip summary",
        );

      expect(summary).toHaveTextContent(
        "Summary incomplete",
      );

      expect(summary).toHaveTextContent(
        "Enter current odds for every staged wager.",
      );

      expect(summary).toHaveTextContent(
        "Enter a proposed stake for every single wager.",
      );
    },
  );

  it(
    "shows quoted parlay payout and correlation caveat",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: 500,
        }),
        gameLeg({
          gameId:
            "2026_01_BUF_MIA",
          awayTeam: "BUF",
          homeTeam: "MIA",
          americanOdds: 150,
          side: "home",
          proposedStake: 900,
        }),
      ]);

      storeMode("parlay");
      renderPanel();

      const summary =
        screen.getByLabelText(
          "Bet slip summary",
        );

      expect(summary).toHaveTextContent(
        "Parlay Summary",
      );

      expect(summary).toHaveTextContent(
        "+377",
      );

      expect(summary).toHaveTextContent(
        "$119.32",
      );

      expect(summary).toHaveTextContent(
        "+$94.32",
      );

      expect(summary).toHaveTextContent(
        "Parlay correlation is not modeled.",
      );

      expect(summary).toHaveTextContent(
        "Combined model probability, expected value, and Kelly sizing are unavailable.",
      );
    },
  );

  it(
    "updates the separate parlay stake",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: null,
        }),
        gameLeg({
          gameId:
            "2026_01_BUF_MIA",
          awayTeam: "BUF",
          homeTeam: "MIA",
          americanOdds: 150,
          side: "home",
          proposedStake: null,
        }),
      ]);

      storeMode("parlay");
      renderPanel();

      fireEvent.change(
        screen.getByLabelText(
          "Parlay stake",
        ),
        {
          target: {
            value: "100",
          },
        },
      );

      const summary =
        screen.getByLabelText(
          "Bet slip summary",
        );

      expect(summary).toHaveTextContent(
        "$477.27",
      );

      expect(summary).toHaveTextContent(
        "+$377.27",
      );
    },
  );

  it(
    "blocks parlay output when one leg is unpriced",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: null,
        }),
        propLeg(),
      ]);

      storeMode("parlay");
      renderPanel();

      const summary =
        screen.getByLabelText(
          "Bet slip summary",
        );

      expect(summary).toHaveTextContent(
        "Enter current odds for every staged wager.",
      );

      expect(summary).toHaveTextContent(
        "Unavailable",
      );
    },
  );

  it(
    "states that Gridiron Edge does not place wagers",
    () => {
      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: 25,
        }),
      ]);

      renderPanel();

      expect(
        screen.getByText(
          /does not place sportsbook wagers/,
        ),
      ).toBeInTheDocument();

      expect(
        screen.queryByRole(
          "button",
          {
            name: /Place Bet/i,
          },
        ),
      ).not.toBeInTheDocument();
    },
  );

  it(
    "retains the clear-slip action",
    async () => {
      const user = userEvent.setup();

      storeLegs([
        gameLeg({
          gameId:
            "2026_01_KC_LAC",
          awayTeam: "KC",
          homeTeam: "LAC",
          americanOdds: -110,
          side: "away",
          proposedStake: 25,
        }),
      ]);

      renderPanel();

      await user.click(
        screen.getByRole("button", {
          name: "Clear Slip",
        }),
      );

      expect(
        screen.getByLabelText(
          "Empty Bet Slip",
        ),
      ).toHaveTextContent(
        "Your Bet Slip is empty.",
      );
    },
  );

  it(
    "explains the empty-slip workflow",
    () => {
      renderPanel();

      const emptyState =
        screen.getByLabelText(
          "Empty Bet Slip",
        );

      expect(emptyState).toHaveTextContent(
        "Add a model edge from Available Edges",
      );

      expect(emptyState).toHaveTextContent(
        "Enter or adjust current odds and a proposed stake",
      );

      expect(emptyState).toHaveTextContent(
        "Props may require manual current odds",
      );

      expect(emptyState).toHaveTextContent(
        "does not place sportsbook wagers",
      );
    },
  );

  it(
    "exposes pressed state for wager and bankroll modes",
    async () => {
      const user = userEvent.setup();

      renderPanel();

      const singleButton =
        screen.getByRole("button", {
          name: "Single",
        });

      const parlayButton =
        screen.getByRole("button", {
          name: "Parlay",
        });

      const trackedButton =
        screen.getByRole("button", {
          name: "Tracked",
        });

      const whatIfButton =
        screen.getByRole("button", {
          name: "What-if",
        });

      expect(singleButton).toHaveAttribute(
        "aria-pressed",
        "true",
      );

      expect(parlayButton).toHaveAttribute(
        "aria-pressed",
        "false",
      );

      expect(trackedButton).toHaveAttribute(
        "aria-pressed",
        "true",
      );

      expect(whatIfButton).toHaveAttribute(
        "aria-pressed",
        "false",
      );

      await user.click(whatIfButton);

      expect(
        sizing.setBankrollMode,
      ).toHaveBeenCalledWith(
        "what-if",
      );
    },
  );

});
