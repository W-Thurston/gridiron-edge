import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TeamMark } from "./TeamMark";
import { TestWrapper } from "../../test/testWrapper";

// Mock the useTeamByAbbr hook
vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(),
}));

import { useTeamByAbbr } from "../../api/team_metadata_hook";

describe("TeamMark", () => {
  it("renders team abbreviation", () => {
    vi.mocked(useTeamByAbbr).mockReturnValue(null);
    render(
      <TestWrapper>
        <TeamMark abbr="KAN" />
      </TestWrapper>,
    );
    expect(screen.getByText("KAN")).toBeInTheDocument();
  });

  it("uses team primary color when available", () => {
    vi.mocked(useTeamByAbbr).mockReturnValue({
      abbr: "KAN",
      name: "Kansas City Chiefs",
      primary_color: "#E31837",
      secondary_color: "#FFB81C",
    } as never);

    render(
      <TestWrapper>
        <TeamMark abbr="KAN" />
      </TestWrapper>,
    );

    const mark = screen.getByText("KAN");
    // Style is applied inline; check background
    expect(mark).toHaveStyle({ background: "#E31837" });
  });

  it("falls back to grey when team not in cache", () => {
    vi.mocked(useTeamByAbbr).mockReturnValue(null);
    render(
      <TestWrapper>
        <TeamMark abbr="KAN" />
      </TestWrapper>,
    );

    const mark = screen.getByText("KAN");
    expect(mark).toHaveStyle({ background: "var(--bg-3)" });
  });

  it("respects size prop", () => {
    vi.mocked(useTeamByAbbr).mockReturnValue(null);
    render(
      <TestWrapper>
        <TeamMark abbr="KAN" size={44} />
      </TestWrapper>,
    );

    const mark = screen.getByText("KAN");
    expect(mark).toHaveStyle({ width: "44px", height: "44px" });
  });
});
