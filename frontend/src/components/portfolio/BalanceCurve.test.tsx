import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BalanceCurve } from "./BalanceCurve";

describe("BalanceCurve", () => {
  it("renders the opening bankroll as a dashed reference line", () => {
    render(
      <BalanceCurve
        points={[
          { timestamp: "2026-09-07T20:47:00Z", bankroll: 60.7 },
          { timestamp: "2026-09-17T00:49:00Z", bankroll: 41.96 },
        ]}
      />,
    );

    expect(screen.getByText("Starting balance · $60.70")).toBeInTheDocument();
    expect(screen.getByTestId("starting-balance-line")).toHaveAttribute(
      "stroke-dasharray",
      "5 4",
    );
    expect(screen.getByRole("img")).toHaveAccessibleName(
      "Balance curve. Starting balance $60.70. Current balance $41.96.",
    );
  });

  it("renders an unavailable marker without valid points", () => {
    render(<BalanceCurve points={[]} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
