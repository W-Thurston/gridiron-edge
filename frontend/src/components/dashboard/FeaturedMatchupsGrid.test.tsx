import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FeaturedMatchupsGrid } from "./FeaturedMatchupsGrid";
import { TestWrapper } from "../../test/testWrapper";

// Mock the API hooks
vi.mock("../../api/hooks", () => ({
  useEdges: vi.fn(() => ({
    data: undefined,
    isLoading: true,
    error: null,
  })),
  useGamesList: vi.fn(() => ({
    data: undefined,
    isLoading: true,
    error: null,
  })),
}));

vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

describe("FeaturedMatchupsGrid", () => {
  it("renders header", () => {
    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );
    expect(screen.getByText("Featured Matchups")).toBeInTheDocument();
  });

  it("renders loading state initially", () => {
    render(
      <TestWrapper>
        <FeaturedMatchupsGrid />
      </TestWrapper>,
    );
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });
});
