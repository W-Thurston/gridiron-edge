import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BlockedScreen } from "./BlockedScreen";
import { TestWrapper } from "../../test/testWrapper";

describe("BlockedScreen", () => {
  it("renders title, description, blocker, and roadmap", () => {
    render(
      <TestWrapper>
        <BlockedScreen
          title="Test Feature"
          description="A test description."
          blocker="test_blocker"
          roadmap="W99"
          requirements={["Requirement one", "Requirement two"]}
        />
      </TestWrapper>,
    );

    expect(screen.getByText("Test Feature")).toBeInTheDocument();
    expect(screen.getByText("A test description.")).toBeInTheDocument();
    expect(screen.getByText(/test_blocker/)).toBeInTheDocument();
    expect(screen.getByText(/W99/)).toBeInTheDocument();
    expect(screen.getByText("Requirement one")).toBeInTheDocument();
    expect(screen.getByText("Requirement two")).toBeInTheDocument();
  });
});
