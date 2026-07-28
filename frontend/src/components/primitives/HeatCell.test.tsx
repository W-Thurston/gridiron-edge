import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HeatCell } from "./HeatCell";

vi.mock("../field-status/usePendingHighlight", () => ({
  usePendingHighlight: () => ({}),
}));

describe("HeatCell", () => {
  it("renders a whole percentage", () => {
    render(<HeatCell value={0.7762} label="Make playoffs" />);

    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Make playoffs: 77.6%"),
    ).toBeInTheDocument();
  });

  it("renders zero distinctly", () => {
    render(<HeatCell value={0} label="Win Super Bowl" />);

    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Win Super Bowl: 0.0%"),
    ).toBeInTheDocument();
  });

  it("renders a positive sub-one-percent probability as less than one percent", () => {
    render(<HeatCell value={0.0012} label="Win Super Bowl" />);

    expect(screen.getByText("<1%")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Win Super Bowl: 0.12%"),
    ).toBeInTheDocument();
  });

  it("renders one as one hundred percent", () => {
    render(<HeatCell value={1} label="Make playoffs" />);

    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("renders pending status through PendingField", () => {
    render(
      <HeatCell
        value={null}
        label="Make playoffs"
        status="pending"
      />,
    );

    expect(screen.getByTitle("Coming soon")).toBeInTheDocument();
  });

  it("renders blocked status through BlockedField", () => {
    render(
      <HeatCell
        value={null}
        label="Make playoffs"
        status={{
          status: "blocked",
          blocker: "no_projections_data",
          roadmap: "data",
        }}
      />,
    );

    expect(
      screen.getByTitle("Not available: no_projections_data (data)"),
    ).toBeInTheDocument();
  });

  it("renders an explicit unavailable fallback when status metadata is absent", () => {
    render(<HeatCell value={null} label="Make playoffs" />);

    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Make playoffs: not available"),
    ).toBeInTheDocument();
  });

  it("clamps visual intensity for values outside the normalized range", () => {
    const { rerender } = render(
      <HeatCell value={-0.2} label="Make playoffs" />,
    );

    expect(screen.getByText("-20%")).toHaveStyle({
      background:
        "color-mix(in oklab, var(--pos) 4%, transparent)",
    });

    rerender(<HeatCell value={1.2} label="Make playoffs" />);

    expect(screen.getByText("120%")).toHaveStyle({
      background:
        "color-mix(in oklab, var(--pos) 34%, transparent)",
    });
  });
});
