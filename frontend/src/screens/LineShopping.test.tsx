import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { useLines } from "../api/hooks";
import { TestWrapper } from "../test/testWrapper";
import { LineShopping } from "./LineShopping";

vi.mock("../api/hooks", () => ({ useLines: vi.fn() }));

const response: components["schemas"]["LineShoppingList"] = {
  _meta: null,
  season: "2026-2027",
  week: 1,
  market: "spread" as const,
  total: 1,
  sportsbooks: ["betmgm", "draftkings", "fanduel"],
  market_fetched_at: ["2026-08-05T22:05:33Z"],
  items: [
    {
      game_id: "2026_01_NE_SEA",
      season: "2026-2027",
      week: 1,
      game_date: "2026-09-09",
      away_team: "New England Patriots",
      home_team: "Seattle Seahawks",
      commence_time: "2026-09-10T00:15:00Z",
      guidance: [
        {
          side: "away",
          model_status: "available",
          model_value: 1.5,
          playable_line: 2.3,
          reference_odds: -110,
          fair_american_odds: null,
          product_id: "weekly-product",
          product_run_id: "weekly-run",
        },
        {
          side: "home",
          model_status: "available",
          model_value: -1.5,
          playable_line: -0.8,
          reference_odds: -110,
          fair_american_odds: null,
          product_id: "weekly-product",
          product_run_id: "weekly-run",
        },
      ],
      offers: [
        offer("draftkings", "away", 3.5, -110, false, true),
        offer("fanduel", "away", 4.5, -120, true, true),
        offer("draftkings", "home", -3.5, -110, true, true),
        offer("fanduel", "home", -4.5, 100, false, true),
      ],
    },
  ],
};

function offer(
  sportsbook: string,
  side: "away" | "home",
  line: number,
  american_odds: number,
  is_best_line: boolean,
  is_best_price: boolean,
) {
  return {
    provider: "the_odds_api",
    provider_event_id: `event-${sportsbook}`,
    sportsbook,
    sportsbook_updated_at: "2026-08-05T22:05:03Z",
    market_fetched_at: "2026-08-05T22:05:33Z",
    commence_time: "2026-09-10T00:15:00Z",
    is_live: false,
    market: "spread" as const,
    side,
    line,
    american_odds,
    is_best_line,
    is_best_price,
    model_status: "available" as const,
    model_value: -1.5,
    model_probability: side === "away" ? 0.58 : 0.42,
    expected_value: side === "away" ? 0.07 : -0.02,
    is_model_approved: side === "away",
    is_best_model_approved_offer: sportsbook === "fanduel" && side === "away",
    product_id: "weekly-product",
    product_run_id: "weekly-run",
  };
}

function loaded(data = response) {
  vi.mocked(useLines).mockReturnValue({
    data,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

describe("LineShopping", () => {
  beforeEach(() => {
    localStorage.clear();
    loaded();
  });

  it("renders current offers with truthful partial coverage", () => {
    render(<TestWrapper><LineShopping /></TestWrapper>);

    expect(screen.getByText("Line Shopping")).toBeInTheDocument();
    expect(screen.getByText("Week 1 · 3 available sportsbooks")).toBeInTheDocument();
    expect(screen.getByText("BetMGM")).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable")).toHaveLength(2);
    expect(screen.getByText("+4.5")).toBeInTheDocument();
    expect(screen.getByText("-120")).toBeInTheDocument();
    expect(screen.queryByText("Best line")).not.toBeInTheDocument();
    expect(screen.queryByText("Best price")).not.toBeInTheDocument();
    expect(screen.getByText("Model New England Patriots +1.5")).toBeInTheDocument();
    expect(screen.getByText("Playable New England Patriots +2.3 or more at -110")).toBeInTheDocument();
        expect(screen.getByText("WED · SEP 9 · 8:15 PM ET")).toBeInTheDocument();
    expect(screen.getByText("Orange price")).toHaveClass("line-shopping-best-price");
    expect(screen.getByRole("button", { name: /highlight model guidance/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("requests the selected market", async () => {
    const user = userEvent.setup();
    render(<TestWrapper><LineShopping /></TestWrapper>);

    expect(useLines).toHaveBeenCalledWith({ market: "spread" });
    await user.click(screen.getByRole("button", { name: "Total" }));
    expect(useLines).toHaveBeenLastCalledWith({ market: "total" });
  });

  it("applies persisted sportsbook selection before building columns", () => {
    localStorage.setItem("hm-app", JSON.stringify({
      sportsbookMode: "selected",
      selectedSportsbooks: ["fanduel"],
    }));

    render(<TestWrapper><LineShopping /></TestWrapper>);

    expect(screen.getByText("FanDuel")).toBeInTheDocument();
    expect(screen.queryByText("DraftKings")).not.toBeInTheDocument();
    expect(screen.queryByText("BetMGM")).not.toBeInTheDocument();
    expect(screen.getByText("Week 1 · 1 available sportsbooks")).toBeInTheDocument();
  });

  it("renders missing-snapshot metadata", () => {
    loaded({
      ...response,
      items: [],
      total: 0,
      sportsbooks: [],
      _meta: {
        field_status: {
          items: {
            status: "blocked" as const,
            blocker: "no_odds_available",
            roadmap: "data",
          },
        },
      },
    });

    render(<TestWrapper><LineShopping /></TestWrapper>);

    expect(screen.getByText("Current sportsbook offers are unavailable.")).toBeInTheDocument();
    expect(screen.getByText("Reason: no_odds_available")).toBeInTheDocument();
  });

  it("distinguishes an analytical empty scope", () => {
    loaded({ ...response, items: [], total: 0, sportsbooks: [] });
    render(<TestWrapper><LineShopping /></TestWrapper>);
    expect(screen.getByText("No current spread offers for this season and week.")).toBeInTheDocument();
  });

  it("renders a labeled focusable comparison region", () => {
    render(<TestWrapper><LineShopping /></TestWrapper>);
    expect(screen.getByRole("region", { name: "Line Shopping" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("table")).toHaveAccessibleName(/current spread offers/i);
  });
});


  it("persists and disables the visual guidance overlay", async () => {
    const user = userEvent.setup();
    const { container } = render(<TestWrapper><LineShopping /></TestWrapper>);
    const toggle = screen.getByRole("button", { name: /highlight model guidance/i });

    expect(container.querySelectorAll(".line-shopping-offer--approved").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".line-shopping-offer--preferred").length).toBe(1);
    expect(container.querySelectorAll(".line-shopping-offer--best-line").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".line-shopping-best-price").length).toBeGreaterThan(0);

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(container.querySelector(".line-shopping-offer--approved")).toBeNull();
    expect(container.querySelector(".line-shopping-offer--preferred")).toBeNull();
    expect(container.querySelector(".line-shopping-offer--best-line")).toBeNull();
    expect(container.querySelector(".line-shopping-best-price")).toBeNull();
    expect(JSON.parse(localStorage.getItem("hm-app") ?? "{}").lineShoppingHighlights).toBe(false);
  });

  it("loads a persisted disabled overlay and keeps guidance text", () => {
    localStorage.setItem("hm-app", JSON.stringify({ lineShoppingHighlights: false }));
    const { container } = render(<TestWrapper><LineShopping /></TestWrapper>);

    expect(screen.getByRole("button", { name: /highlight model guidance/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(container.querySelector(".line-shopping-offer--approved")).toBeNull();
    expect(screen.getByText("Playable New England Patriots +2.3 or more at -110")).toBeInTheDocument();
    expect(screen.queryByText("Model approved")).not.toBeInTheDocument();
  });

  it("keeps market classifications in the offer explanation", async () => {
    const user = userEvent.setup();
    render(<TestWrapper><LineShopping /></TestWrapper>);
    const trigger = screen.getByRole("button", {
      name: /explain New England Patriots \+4\.5 at -120/i,
    });

    await user.hover(trigger);

    const tooltip = screen.getByRole("tooltip");
    expect(
      within(tooltip).getByText(/This is the best available line\./),
    ).toBeInTheDocument();
    expect(
      within(tooltip).getByText(/This is the best price available at this exact line\./),
    ).toBeInTheDocument();
    expect(
      within(tooltip).getByText(/This is a preferred model-approved offer\./),
    ).toBeInTheDocument();
  });


it("explains an exact sportsbook offer on hover", async () => {
  const user = userEvent.setup();
  render(<TestWrapper><LineShopping /></TestWrapper>);
  const trigger = screen.getByRole("button", {
    name: /explain New England Patriots \+3\.5 at -110/i,
  });

  await user.hover(trigger);

  const tooltip = screen.getByRole("tooltip");
  expect(within(tooltip).getByText("Bet outcome")).toBeInTheDocument();
  expect(within(tooltip).getByText(/wins if New England Patriots wins or loses by 3 points or fewer/)).toBeInTheDocument();
  expect(within(tooltip).getByText(/a \$110 stake would produce \$100 in profit/)).toBeInTheDocument();
  expect(within(tooltip).getByText(/Expected value: \+7\.0%/)).toBeInTheDocument();
});

it("explains side-oriented model guidance on keyboard focus", async () => {
  const user = userEvent.setup();
  render(<TestWrapper><LineShopping /></TestWrapper>);
  const trigger = screen.getByRole("button", {
    name: /explain New England Patriots spread model guidance/i,
  });

  await user.click(trigger);

  const tooltip = screen.getByRole("tooltip");
  expect(within(tooltip).getByText(/-3\.5 is larger and more favorable than -4\.5/)).toBeInTheDocument();
  expect(trigger).toHaveAttribute("aria-expanded", "true");
  await user.tab();
});
