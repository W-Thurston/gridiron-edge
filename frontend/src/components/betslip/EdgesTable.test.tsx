import {
  render,
  screen,
} from "@testing-library/react";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { EdgesTable } from "./EdgesTable";
import { TestWrapper } from "../../test/testWrapper";
import { useEdges } from "../../api/hooks";

vi.mock("../../api/hooks", () => ({
  useEdges: vi.fn(),
}));

vi.mock(
  "../../api/team_metadata_hook",
  () => ({
    useTeamByAbbr: vi.fn(
      () => null,
    ),
  }),
);

function mockEmptyEdges() {
  vi.mocked(
    useEdges,
  ).mockReturnValue({
    data: {
      items: [],
      total: 0,
      bankroll: null,
      kelly_multiplier: 0.25,
      _meta: null,
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

function mockPopulatedEdges() {
  vi.mocked(
    useEdges,
  ).mockReturnValue({
    data: {
      season: "2026-2027",
      week: 1,
      min_ev: 0,
      total: 1,
      bankroll: 2500,
      kelly_multiplier: 0.25,
      _meta: null,
      items: [
        {
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
          market_type:
            "moneyline",
          market_value: 0.45,
          model_key:
            "random_forest_win_prob",
          model_value: 0.58,
          point_edge: null,
          side: "away",
        },
      ],
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

describe("EdgesTable sizing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEmptyEdges();
  });

  it(
    "table",
    () => {
      mockPopulatedEdges();

      render(
        <TestWrapper>
          <EdgesTable
            bankroll={2500}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        screen.getByRole("table", {
          name: /Available model edges/,
        }),
      ).toHaveClass(
        "betslip-edges-table",
      );

      expect(
        screen.getByRole("region", {
          name: "Available Edges",
        }),
      ).toHaveClass(
        "betslip-edges-scroll",
      );

      for (const heading of [
        "Matchup",
        "Market",
        "Side",
        "EV",
        "Strength",
      ]) {
        expect(
          screen.getByRole(
            "columnheader",
            { name: heading },
          ),
        ).toBeInTheDocument();
      }
    },
  );

  it(
    "labels the Add action with the wager identity",
    () => {
      mockPopulatedEdges();

      render(
        <TestWrapper>
          <EdgesTable
            bankroll={2500}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        screen.getByRole("button", {
          name:
            "Add moneyline away for KC at LAC to the Bet Slip",
        }),
      ).toBeInTheDocument();
    },
  );

  it(
    "passes an explicit bankroll and multiplier",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={2500}
            kellyMultiplier={0.1}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        bankroll: 2500,
        kelly_multiplier: 0.1,
      });

      expect(
        screen.getByText(
          "No edges available.",
        ),
      ).toBeInTheDocument();

      expect(
        screen.getByText(
          /stage one here for current-price/,
        ),
      ).toBeInTheDocument();
    },
  );

  it(
    "omits bankroll when the sizing basis is unavailable",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={null}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        kelly_multiplier: 0.25,
      });
    },
  );

  it(
    "preserves zero as an explicit bankroll",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={0}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        bankroll: 0,
        kelly_multiplier: 0.25,
      });
    },
  );
});
