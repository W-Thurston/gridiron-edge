import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Tools } from "./Tools";

describe("Tools", () => {
  it("renders all three calculator cards", () => {
    render(<Tools />);
    expect(screen.getByText("Odds Converter")).toBeInTheDocument();
    expect(screen.getByText("Kelly Calculator")).toBeInTheDocument();
    expect(screen.getByText("Payout Calculator")).toBeInTheDocument();
  });

  it("converts -150 American to 1.667 decimal and 60% implied", async () => {
    const user = userEvent.setup();
    render(<Tools />);

    // Find the American Odds input in the Odds Converter card.
    const americanInputs = screen.getAllByDisplayValue("-110");
    const americanInput = americanInputs[0];  // First one is Odds Converter.

    await user.clear(americanInput);
    await user.type(americanInput, "-150");

    expect(screen.getByText("1.667")).toBeInTheDocument();
    expect(screen.getByText("60.0%")).toBeInTheDocument();
  });
});
