import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PropEdgesRail } from "./PropEdgesRail";
import { TestWrapper } from "../../test/testWrapper";

vi.mock("../../api/hooks", () => ({
  usePropsList: vi.fn(),
}));

vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

import { usePropsList } from "../../api/hooks";

describe("PropEdgesRail", () => {
  it("renders header", () => {
    vi.mocked(usePropsList).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <PropEdgesRail />
      </TestWrapper>,
    );
    expect(screen.getByText("Prop Edges")).toBeInTheDocument();
  });

  it("renders loading state", () => {
    vi.mocked(usePropsList).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <PropEdgesRail />
      </TestWrapper>,
    );
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders 'See all →' link", () => {
    vi.mocked(usePropsList).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <PropEdgesRail />
      </TestWrapper>,
    );
    expect(screen.getByText("See all →")).toBeInTheDocument();
  });

  it("renders empty state when no props", () => {
    vi.mocked(usePropsList).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <PropEdgesRail />
      </TestWrapper>,
    );
    expect(screen.getByText(/No prop projections yet/)).toBeInTheDocument();
  });

  it("renders top prop rows sorted by predicted_mean descending", () => {
    vi.mocked(usePropsList).mockReturnValue({
      data: {
        items: [
          {
            prop_id: "1",
            game_id:
              "2026_01_KC_LAC",
            player_id: "player-1",
            player_name: "P. Mahomes",
            position: "QB",
            team: "KAN",
            stat_type:
              "qb_pass_yards",
            model_key:
              "elastic_net_qb_pass_yards",
            projection: {
              predicted_mean: 275.0,
              predicted_std: 45.0,
            },
            line_context: {
              line: null,
              p_over: null,
              lean: null,
              confidence_tier: null,
            },
          },
          {
            prop_id: "2",
            game_id:
              "2026_01_BAL_BUF",
            player_id: "player-2",
            player_name: "L. Jackson",
            position: "QB",
            team: "BAL",
            stat_type:
              "qb_pass_yards",
            model_key:
              "elastic_net_qb_pass_yards",
            projection: {
              predicted_mean: 320.0,
              predicted_std: 40.0,
            },
            line_context: {
              line: null,
              p_over: null,
              lean: null,
              confidence_tier: null,
            },
          },
        ],
        _meta: {},
      },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <PropEdgesRail />
      </TestWrapper>,
    );

    // L. Jackson has higher predicted_mean, so should appear
    expect(screen.getByText("L. Jackson")).toBeInTheDocument();
    expect(screen.getByText("P. Mahomes")).toBeInTheDocument();
  });
});
