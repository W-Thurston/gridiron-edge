import {
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  describe,
  expect,
  it,
  vi,
} from "vitest";
import type { components } from "../../api/schema";
import {
  createGameBetLeg,
  createPropBetLeg,
  type BetLeg,
} from "../../utils/betLegs";
import { TestWrapper } from "../../test/testWrapper";
import { BetLegCard } from "./BetLegCard";

type EdgeApiRow =
  components["schemas"]["EdgeRow"];

type PropApi =
  components["schemas"]["PropSummary"];

const ADDED_AT =
  "2026-07-29T20:00:00.000Z";

function edge(
  overrides: Partial<EdgeApiRow> = {},
): EdgeApiRow {
  return {
    american_odds: -110,
    away_team: "KC",
    cover_prob: null,
    edge_strength: "strong",
    ev: 0.08,
    game_id:
      "2026_01_KC_LAC",
    home_team: "LAC",
    kelly_frac: 0.08,
    kelly_stake: 20,
    market_type: "moneyline",
    market_value: 0.45,
    model_key:
      "random_forest_win_prob",
    model_value: 0.58,
    point_edge: null,
    side: "away",
    ...overrides,
  };
}

function prop(): PropApi {
  return {
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
}

type RenderCardOptions = {
  leg?: BetLeg;
  bankroll?: number | null;
  kellyMultiplier?: number;
};

function renderCard({
  leg = createGameBetLeg({
    edge: edge(),
    source: "betslip-edges",
    addedAt: ADDED_AT,
    referenceBankroll: 2500,
    referenceKellyMultiplier:
      0.1,
  }),
  bankroll = 2500,
  kellyMultiplier = 0.1,
}: RenderCardOptions = {}) {
  const onUpdateCurrentOdds =
    vi.fn();
  const onUpdateProposedStake =
    vi.fn();
  const onUpdateSportsbook =
    vi.fn();
  const onUpdateNote = vi.fn();
  const onRemove = vi.fn();

  render(
    <TestWrapper>
      <BetLegCard
        leg={leg}
        oddsFormat="american"
        bankroll={bankroll}
        kellyMultiplier={
          kellyMultiplier
        }
        onUpdateCurrentOdds={
          onUpdateCurrentOdds
        }
        onUpdateProposedStake={
          onUpdateProposedStake
        }
        onUpdateSportsbook={
          onUpdateSportsbook
        }
        onUpdateNote={onUpdateNote}
        onRemove={onRemove}
      />
    </TestWrapper>,
  );

  return {
    onUpdateCurrentOdds,
    onUpdateProposedStake,
    onUpdateSportsbook,
    onUpdateNote,
    onRemove,
  };
}

describe("BetLegCard", () => {
  it(
    "shows immutable reference and current game analysis",
    () => {
      renderCard();

      expect(
        screen.getByText("-110"),
      ).toBeInTheDocument();

      expect(
        screen.getByText("+8.0%"),
      ).toBeInTheDocument();

      expect(
        screen.getByText(
          "Positive modeled EV",
        ),
      ).toBeInTheDocument();

      expect(
        screen.getByText("$29.50"),
      ).toBeInTheDocument();
    },
  );

  it(
    "updates current American odds",
    () => {
        const {
        onUpdateCurrentOdds,
        } = renderCard();

        fireEvent.change(
        screen.getByLabelText(
            "Current American odds",
        ),
        {
            target: {
            value: "-125",
            },
        },
        );

        expect(
        onUpdateCurrentOdds,
        ).toHaveBeenCalledWith(-125);
    },
    );

  it(
    "clears current American odds",
    () => {
        const {
        onUpdateCurrentOdds,
        } = renderCard();

        fireEvent.change(
        screen.getByLabelText(
            "Current American odds",
        ),
        {
            target: {
            value: "",
            },
        },
        );

        expect(
        onUpdateCurrentOdds,
        ).toHaveBeenCalledWith(null);
    },
    );

  it(
    "updates proposed stake",
    () => {
        const {
        onUpdateProposedStake,
        } = renderCard();

        fireEvent.change(
        screen.getByLabelText(
            "Proposed stake",
        ),
        {
            target: {
            value: "50",
            },
        },
        );

        expect(
        onUpdateProposedStake,
        ).toHaveBeenCalledWith(50);
    },
    );

  it(
    "shows unpriced prop reference state without fabricating odds",
    () => {
      const leg = createPropBetLeg({
        prop: prop(),
        side: "over",
        source:
          "dashboard-prop-edges",
        addedAt: ADDED_AT,
      });

      renderCard({ leg });

      expect(
        screen.getByText(
          "Reference price unavailable",
        ),
      ).toBeInTheDocument();

      expect(
        screen.getByText(
            "Threshold unavailable",
        ),
        ).toBeInTheDocument();

        expect(
        screen.getByText("-156"),
        ).toBeInTheDocument();

      expect(
        screen.getByLabelText(
          "Current American odds",
        ),
      ).toHaveValue(null);
    },
  );

  it(
    "shows unavailable dollar sizing without bankroll",
    () => {
        renderCard({
        bankroll: null,
        });

        const analysisSection =
        screen.getByLabelText(
            "Model analysis",
        );

        expect(
        within(
            analysisSection,
        ).getByText("Unavailable"),
        ).toBeInTheDocument();
    },
    );

  it(
    "updates optional sportsbook and note",
    async () => {
        const user = userEvent.setup();
        const {
        onUpdateSportsbook,
        onUpdateNote,
        } = renderCard();

        await user.click(
        screen.getByText(
            "Draft details",
        ),
        );

        fireEvent.change(
        screen.getByPlaceholderText(
            "Optional manual entry",
        ),
        {
            target: {
            value: "DraftKings",
            },
        },
        );

        fireEvent.change(
        screen.getByPlaceholderText(
            "Optional draft note",
        ),
        {
            target: {
            value:
                "Check before kickoff",
            },
        },
        );

        expect(
        onUpdateSportsbook,
        ).toHaveBeenCalledWith(
        "DraftKings",
        );

        expect(
        onUpdateNote,
        ).toHaveBeenCalledWith(
        "Check before kickoff",
        );
    },
    );

  it(
    "removes the staged leg",
    async () => {
      const user = userEvent.setup();
      const { onRemove } =
        renderCard();

      await user.click(
        screen.getByRole("button", {
          name: /Remove KC at LAC/,
        }),
      );

      expect(
        onRemove,
      ).toHaveBeenCalledTimes(1);
    },
  );
});
