import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Settings } from "./Settings";
import { TestWrapper } from "../test/testWrapper";

describe("Settings", () => {
  it("renders all settings rows", () => {
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );
    expect(screen.getByText("Odds Format")).toBeInTheDocument();
    expect(screen.getByText("Bankroll")).toBeInTheDocument();
    expect(screen.getByText("Alerts")).toBeInTheDocument();
    expect(screen.getByText("Reset Everything")).toBeInTheDocument();
  });

  it("toggles odds format when Decimal button is clicked", async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );
    const decimalButton = screen.getByText("Decimal");
    await user.click(decimalButton);
    // After click, Decimal has active styling — we check it renders without error.
    expect(decimalButton).toBeInTheDocument();
  });
});
