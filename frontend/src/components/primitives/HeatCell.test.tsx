import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HeatCell } from "./HeatCell";

vi.mock("../field-status/usePendingHighlight", () => ({
  usePendingHighlight: () => ({}),
}));

function renderHeatCell(
  props: React.ComponentProps<typeof HeatCell>,
) {
  return render(
    <table>
      <tbody>
        <tr>
          <HeatCell {...props} />
        </tr>
      </tbody>
    </table>,
  );
}

describe("HeatCell", () => {
  it("renders a whole percentage", () => {
    renderHeatCell({
      value: 0.7762,
      label: "Make playoffs",
    });

    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(
      screen.getByRole("cell", {
        name: "Make playoffs: 77.6%",
      }),
    ).toBeInTheDocument();
  });

  it("renders zero distinctly", () => {
    renderHeatCell({
      value: 0,
      label: "Win Super Bowl",
    });

    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(
      screen.getByRole("cell", {
        name: "Win Super Bowl: 0.0%",
      }),
    ).toBeInTheDocument();
  });

  it("renders a positive sub-one-percent probability as less than one percent", () => {
    renderHeatCell({
      value: 0.0012,
      label: "Win Super Bowl",
    });

    expect(screen.getByText("<1%")).toBeInTheDocument();
    expect(
      screen.getByRole("cell", {
        name: "Win Super Bowl: 0.12%",
      }),
    ).toBeInTheDocument();
  });

  it("renders one as one hundred percent", () => {
    renderHeatCell({
      value: 1,
      label: "Make playoffs",
    });

    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("renders blocked status through BlockedField", () => {
    renderHeatCell({
      value: null,
      label: "Make playoffs",
      status: {
        status: "blocked",
        blocker: "no_projections_data",
        roadmap: "data",
      },
    });

    expect(
      screen.getByTitle("Not available: no_projections_data (data)"),
    ).toBeInTheDocument();
  });

  it("renders an explicit unavailable fallback when status metadata is absent", () => {
    renderHeatCell({
      value: null,
      label: "Make playoffs",
    });

    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(
      screen.getByRole("cell", {
        name: "Make playoffs: not available",
      }),
    ).toBeInTheDocument();
  });

  it("clamps visual intensity for values outside the normalized range", () => {
    const { rerender } = render(
      <table>
        <tbody>
          <tr>
            <HeatCell value={-0.2} label="Make playoffs" />
          </tr>
        </tbody>
      </table>,
    );

    expect(
      screen.getByRole("cell", {
        name: "Make playoffs: -20.00%",
      }),
    ).toHaveStyle({
      background:
        "color-mix(in oklab, var(--pos) 4%, transparent)",
    });

    rerender(
      <table>
        <tbody>
          <tr>
            <HeatCell value={1.2} label="Make playoffs" />
          </tr>
        </tbody>
      </table>,
    );

    expect(
      screen.getByRole("cell", {
        name: "Make playoffs: 120.0%",
      }),
    ).toHaveStyle({
      background:
        "color-mix(in oklab, var(--pos) 34%, transparent)",
    });
  });
});
