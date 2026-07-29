import {
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import {
  beforeEach,
  describe,
  expect,
  it,
} from "vitest";
import type { components } from "../api/schema";
import {
  createGameBetLeg,
  createPropBetLeg,
  type BetLeg,
} from "../utils/betLegs";
import {
  BetSlipProvider,
  useBetSlip,
} from "./BetSlipContext";

type EdgeApiRow =
  components["schemas"]["EdgeRow"];

type PropApi =
  components["schemas"]["PropSummary"];

const LEGS_KEY = "hm-betslip-v2";
const MODE_KEY = "hm-betslip-mode-v2";
const ADDED_AT =
  "2026-07-29T16:00:00.000Z";

function edge(): EdgeApiRow {
  return {
    american_odds: -110,
    away_team: "KC",
    confidence_tier: "High",
    cover_prob: null,
    edge_strength: "strong",
    ev: 0.08,
    game_id: "2026_01_KC_LAC",
    home_team: "LAC",
    kelly_frac: 0.08,
    kelly_stake: 20,
    market_type: "moneyline",
    market_value: 0.45,
    model_key: "random_forest_win_prob",
    model_value: 0.58,
    point_edge: null,
    side: "away",
  };
}

function prop(): PropApi {
  return {
    game_id: "2026_01_KC_LAC",
    line_context: {
      line: 274.5,
      p_over: 0.61,
      lean: "Over",
      confidence_tier: "Moderate",
    },
    model_key:
      "elastic_net_qb_pass_yards",
    player_id: "00-0033873",
    player_name: "Patrick Mahomes",
    position: "QB",
    projection: {
      predicted_mean: 289.4,
      predicted_std: 71.2,
    },
    prop_id:
      "2026_01_KC_LAC__00-0033873__qb_pass_yards",
    stat_type: "qb_pass_yards",
    team: "KC",
  };
}

function gameLeg() {
  return createGameBetLeg({
    edge: edge(),
    source: "betslip-edges",
    addedAt: ADDED_AT,
    referenceBankroll: 2500,
    referenceKellyMultiplier: 0.1,
  });
}

function propLeg() {
  return createPropBetLeg({
    prop: prop(),
    side: "over",
    source: "dashboard-prop-edges",
    addedAt: ADDED_AT,
  });
}

function Harness() {
  const {
    legs,
    mode,
    add,
    remove,
    clear,
    setMode,
  } = useBetSlip();

  return (
    <div>
      <div data-testid="count">
        {legs.length}
      </div>

      <div data-testid="mode">
        {mode}
      </div>

      <div data-testid="ids">
        {legs
          .map((leg) => leg.id)
          .join("|")}
      </div>

      <button
        type="button"
        onClick={() => add(gameLeg())}
      >
        Add game
      </button>

      <button
        type="button"
        onClick={() => add(propLeg())}
      >
        Add prop
      </button>

      <button
        type="button"
        onClick={() =>
          add(
            {
              ...gameLeg(),
              id: "invalid",
            } as BetLeg,
          )
        }
      >
        Add invalid
      </button>

      <button
        type="button"
        onClick={() =>
          remove(gameLeg().id)
        }
      >
        Remove game
      </button>

      <button
        type="button"
        onClick={clear}
      >
        Clear
      </button>

      <button
        type="button"
        onClick={() => setMode("parlay")}
      >
        Parlay
      </button>
    </div>
  );
}

function renderProvider(
  children: ReactNode = <Harness />,
) {
  return render(
    <BetSlipProvider>
      {children}
    </BetSlipProvider>,
  );
}

describe("BetSlipContext", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("defaults to an empty single slip", () => {
    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");

    expect(
      screen.getByTestId("mode"),
    ).toHaveTextContent("single");
  });

  it("loads valid v2 legs", () => {
    localStorage.setItem(
      LEGS_KEY,
      JSON.stringify([
        gameLeg(),
        propLeg(),
      ]),
    );

    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("2");
  });

  it("ignores legacy prototype storage", () => {
    localStorage.setItem(
      "hm-betslip",
      JSON.stringify([
        {
          id: "legacy",
          odds: -110,
        },
      ]),
    );

    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");
  });

  it("drops malformed stored legs individually", () => {
    localStorage.setItem(
      LEGS_KEY,
      JSON.stringify([
        gameLeg(),
        {
          ...propLeg(),
          version: 1,
        },
      ]),
    );

    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("1");
  });

  it("recovers from invalid JSON", () => {
    localStorage.setItem(
      LEGS_KEY,
      "{not-json",
    );

    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");
  });

  it("rejects non-array stored state", () => {
    localStorage.setItem(
      LEGS_KEY,
      JSON.stringify(gameLeg()),
    );

    renderProvider();

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");
  });

  it("loads valid mode and rejects invalid mode", () => {
    localStorage.setItem(
      MODE_KEY,
      "parlay",
    );

    const first = renderProvider();

    expect(
      screen.getByTestId("mode"),
    ).toHaveTextContent("parlay");

    first.unmount();

    localStorage.setItem(
      MODE_KEY,
      "invalid",
    );

    renderProvider();

    expect(
      screen.getByTestId("mode"),
    ).toHaveTextContent("single");
  });

  it("adds valid game and prop legs", async () => {
    const user = userEvent.setup();
    renderProvider();

    await user.click(
      screen.getByRole("button", {
        name: "Add game",
      }),
    );

    await user.click(
      screen.getByRole("button", {
        name: "Add prop",
      }),
    );

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("2");
  });

  it("rejects invalid runtime legs", async () => {
    const user = userEvent.setup();
    renderProvider();

    await user.click(
      screen.getByRole("button", {
        name: "Add invalid",
      }),
    );

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");
  });

  it("deduplicates canonical IDs", async () => {
    const user = userEvent.setup();
    renderProvider();

    await user.click(
      screen.getByRole("button", {
        name: "Add game",
      }),
    );

    await user.click(
      screen.getByRole("button", {
        name: "Add game",
      }),
    );

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("1");
  });

  it("removes and clears legs", async () => {
    const user = userEvent.setup();
    renderProvider();

    await user.click(
      screen.getByRole("button", {
        name: "Add game",
      }),
    );

    await user.click(
      screen.getByRole("button", {
        name: "Add prop",
      }),
    );

    await user.click(
      screen.getByRole("button", {
        name: "Remove game",
      }),
    );

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("1");

    await user.click(
      screen.getByRole("button", {
        name: "Clear",
      }),
    );

    expect(
      screen.getByTestId("count"),
    ).toHaveTextContent("0");
  });

  it("persists v2 legs and mode", async () => {
    const user = userEvent.setup();
    renderProvider();

    await user.click(
      screen.getByRole("button", {
        name: "Add game",
      }),
    );

    await user.click(
      screen.getByRole("button", {
        name: "Parlay",
      }),
    );

    expect(
      JSON.parse(
        localStorage.getItem(
          LEGS_KEY,
        ) ?? "[]",
      ),
    ).toHaveLength(1);

    expect(
      localStorage.getItem(MODE_KEY),
    ).toBe("parlay");
  });
});
