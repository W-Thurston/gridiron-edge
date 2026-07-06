import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelEdgesTable } from "./ModelEdgesTable";
import { TestWrapper } from "../../test/testWrapper";

vi.mock("../../api/hooks", () => ({
  useEdges: vi.fn(),
}));

vi.mock("../../api/team_metadata_hook", () => ({
  useTeamByAbbr: vi.fn(() => null),
}));

import { useEdges } from "../../api/hooks";

describe("ModelEdgesTable", () => {
  it("renders header", () => {
    vi.mocked(useEdges).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelEdgesTable />
      </TestWrapper>,
    );
    expect(screen.getByText("Model Edges")).toBeInTheDocument();
  });

  it("renders loading state when data is loading", () => {
    vi.mocked(useEdges).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelEdgesTable />
      </TestWrapper>,
    );
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders 4 filter tabs when data is loaded", () => {
    vi.mocked(useEdges).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelEdgesTable />
      </TestWrapper>,
    );
    expect(screen.getByText("All")).toBeInTheDocument();
    expect(screen.getByText("Spread")).toBeInTheDocument();
    expect(screen.getByText("Total")).toBeInTheDocument();
    expect(screen.getByText("Moneyline")).toBeInTheDocument();
  });

  it("renders empty state when no edges", () => {
    vi.mocked(useEdges).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelEdgesTable />
      </TestWrapper>,
    );
    expect(screen.getByText(/No edges available/)).toBeInTheDocument();
  });
});
