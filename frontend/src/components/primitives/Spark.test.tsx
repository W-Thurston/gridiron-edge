import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Spark } from "./Spark";

describe("Spark", () => {
  it("returns null for empty data", () => {
    const { container } = render(<Spark data={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("returns null for undefined data", () => {
    const { container } = render(<Spark data={undefined as unknown as number[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders SVG for single point", () => {
    const { container } = render(<Spark data={[50]} />);
    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    const circle = container.querySelector("circle");
    expect(circle).toBeInTheDocument();
  });

  it("renders SVG polyline for multiple points", () => {
    const { container } = render(<Spark data={[10, 20, 30, 25]} />);
    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    const path = container.querySelector("path");
    expect(path).toBeInTheDocument();
  });

  it("respects width and height props", () => {
    const { container } = render(<Spark data={[1, 2, 3]} width={200} height={50} />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("width", "200");
    expect(svg).toHaveAttribute("height", "50");
  });

  it("respects color prop", () => {
    const { container } = render(<Spark data={[1, 2, 3]} color="#FF0000" />);
    const path = container.querySelector("path");
    expect(path).toHaveAttribute("stroke", "#FF0000");
  });

  it("handles flat data (all same values) without error", () => {
    const { container } = render(<Spark data={[10, 10, 10]} />);
    const path = container.querySelector("path");
    expect(path).toBeInTheDocument();
  });
});
