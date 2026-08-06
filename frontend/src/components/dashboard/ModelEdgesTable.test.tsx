import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../../api/schema";
import { useEdges } from "../../api/hooks";
import { TestWrapper } from "../../test/testWrapper";
import { ModelEdgesTable } from "./ModelEdgesTable";

vi.mock("../../api/hooks", () => ({ useEdges: vi.fn() }));
vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

type EdgeList = components["schemas"]["EdgeList"];
type EdgeDiagnostics = components["schemas"]["EdgeDiagnosticsResponse"];
type EdgeBlocker = components["schemas"]["EdgeDiagnosticBlocker"];

function diagnostics(
  overrides: Partial<EdgeDiagnostics> = {},
): EdgeDiagnostics {
  return {
    season: "2026-2027",
    week: 1,
    prediction_game_count: 1,
    market_game_count: 1,
    matched_game_count: 1,
    complete_moneyline_count: 1,
    complete_spread_count: 1,
    complete_total_count: 1,
    eligible_market_count: 3,
    calculated_edge_count: 3,
    positive_edge_count: 1,
    filtered_edge_count: 1,
    state: "positive_edges",
    blockers: [],
    ...overrides,
  };
}

function response(overrides: Partial<EdgeList> = {}): EdgeList {
  return {
    diagnostics: diagnostics(),
    items: [],
    total: 0,
    ...overrides,
  };
}

function edge(): components["schemas"]["EdgeRow"] {
  return {
    game_id: "2026_01_KC_LAC",
    away_team: "Kansas City Chiefs",
    home_team: "Los Angeles Chargers",
    model_key: "win_prob_elo",
    market_type: "moneyline",
    side: "home",
    model_value: 0.58,
    market_value: 0.52,
    american_odds: -110,
    cover_prob: 0.61,
    ev: 0.08,
    edge_strength: "strong",
  };
}

function mockLoaded(data: EdgeList) {
  vi.mocked(useEdges).mockReturnValue({
    data,
    isLoading: false,
    error: null,
  } as never);
}

const blockers: EdgeBlocker[] = [
  "no_predictions",
  "no_market_data",
  "market_wrong_scope",
  "market_stale",
  "zero_matched_games",
  "incomplete_markets",
];

describe("ModelEdgesTable", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    vi.mocked(useEdges).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);
  });

  it("renders loading state", () => {
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it.each(blockers)("renders blocker %s", (blocker) => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "blocked",
        blockers: [blocker],
        calculated_edge_count: 0,
        positive_edge_count: 0,
        filtered_edge_count: 0,
      }),
    }));
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    expect(screen.getByRole("status")).toHaveAttribute(
      "data-edge-result-kind",
      "blocked",
    );
  });

  it("distinguishes no positive edges", () => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "no_positive_edges",
        positive_edge_count: 0,
        filtered_edge_count: 0,
      }),
    }));
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    expect(screen.getByText("No positive edges.")).toBeInTheDocument();
  });

  it("distinguishes positive edges removed by minimum EV", () => {
    mockLoaded(response({
      diagnostics: diagnostics({
        state: "positive_edges",
        positive_edge_count: 2,
        filtered_edge_count: 0,
      }),
    }));
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    expect(screen.getByText("No edges passed this filter.")).toBeInTheDocument();
  });

  it("renders real fair and cover probabilities without a synthetic band", () => {
    mockLoaded(response({ items: [edge()], total: 1 }));
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    expect(screen.getByText("58.0%")).toBeInTheDocument();
    expect(screen.getByText("61.0%")).toBeInTheDocument();
    expect(screen.queryByText(/band/i)).not.toBeInTheDocument();
  });

  it("distinguishes an active-tab empty state from service emptiness", async () => {
    const user = userEvent.setup();
    mockLoaded(response({ items: [edge()], total: 1 }));
    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);
    await user.click(screen.getByText("Spread"));
    expect(
      screen.getByText("No positive spread edges in this view."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });


  it("honors a selected sportsbook when rendering model offers", () => {
    localStorage.setItem("hm-app", JSON.stringify({
      sportsbookMode: "selected",
      selectedSportsbooks: ["fanduel"],
    }));
    mockLoaded(response({
      items: [
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-dk",
          sportsbook: "draftkings",
          american_odds: -150,
          ev: 0.08,
        },
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-fd",
          sportsbook: "fanduel",
          american_odds: -140,
          ev: 0.1,
        },
      ],
      total: 2,
    }));

    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);

    expect(screen.queryByText("DraftKings")).not.toBeInTheDocument();
    expect(screen.getByText("FanDuel")).toBeInTheDocument();
    expect(screen.queryByText("-150")).not.toBeInTheDocument();
    expect(screen.getByText("-140")).toBeInTheDocument();
  });



  it("shows one best offer per wager family and expands alternatives", async () => {
    const user = userEvent.setup();
    mockLoaded(response({
      items: [
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-dk",
          sportsbook: "draftkings",
          american_odds: -150,
          ev: 0.08,
        },
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-fd",
          sportsbook: "fanduel",
          american_odds: -140,
          ev: 0.1,
        },
      ],
      total: 2,
    }));

    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);

    expect(screen.getByText("FanDuel")).toBeInTheDocument();
    expect(screen.queryByText("DraftKings")).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", {
      name: "View 1 other offer for moneyline home in Kansas City Chiefs at Los Angeles Chargers",
    });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await user.click(toggle);

    expect(screen.getByText("DraftKings")).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("button", {
        name: "Add DraftKings moneyline home for Kansas City Chiefs at Los Angeles Chargers to the Bet Slip",
      }),
    ).toBeInTheDocument();
  });

  it("applies the top-six limit to wager families instead of raw offers", () => {
    const duplicateOffers = Array.from({ length: 6 }, (_, index) => ({
      ...edge(),
      provider: "the_odds_api",
      provider_event_id: `same-family-${index}`,
      sportsbook: `book-${index}`,
      ev: 0.2 - index * 0.01,
    }));
    const distinctFamilies = Array.from({ length: 6 }, (_, index) => ({
      ...edge(),
      game_id: `game-${index}`,
      provider: "the_odds_api",
      provider_event_id: `other-${index}`,
      sportsbook: "draftkings",
      ev: 0.1 - index * 0.01,
    }));
    mockLoaded(response({
      items: [...duplicateOffers, ...distinctFamilies],
      total: 12,
    }));

    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);

    expect(
      screen.getAllByRole("button", { name: /View details for/ }),
    ).toHaveLength(6);
    expect(screen.getByText("5 other offers")).toBeInTheDocument();
  });

  it("does not offer expansion after selected-book filtering", () => {
    localStorage.setItem("hm-app", JSON.stringify({
      sportsbookMode: "selected",
      selectedSportsbooks: ["fanduel"],
    }));
    mockLoaded(response({
      items: [
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-dk",
          sportsbook: "draftkings",
          ev: 0.08,
        },
        {
          ...edge(),
          provider: "the_odds_api",
          provider_event_id: "event-fd",
          sportsbook: "fanduel",
          ev: 0.1,
        },
      ],
      total: 2,
    }));

    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);

    expect(screen.getByText("FanDuel")).toBeInTheDocument();
    expect(screen.queryByText(/other offer/)).not.toBeInTheDocument();
  });



  it("uses explicit matchup controls instead of interactive table rows", () => {
    mockLoaded(response({ items: [edge()], total: 1 }));

    render(<TestWrapper><ModelEdgesTable /></TestWrapper>);

    expect(
      screen.getByRole("button", {
        name: "View details for Kansas City Chiefs at Los Angeles Chargers",
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("row", {
      name: /Kansas City Chiefs.*Los Angeles Chargers/,
    })).not.toHaveAttribute("role", "button");
    expect(screen.getByRole("row", {
      name: /Kansas City Chiefs.*Los Angeles Chargers/,
    })).not.toHaveAttribute("tabindex");
  });

});
