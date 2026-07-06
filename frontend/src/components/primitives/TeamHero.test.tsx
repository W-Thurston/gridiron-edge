import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TeamHero } from "./TeamHero";

const TEAM = {
  abbr: "KAN",
  city: "Kansas City",
  name: "Chiefs",
  primary_color: "#E31837",
  conference: "AFC",
  division: "W",
};

describe("TeamHero", () => {
  it("renders team abbreviation in mark", () => {
    render(<TeamHero team={TEAM} />);
    expect(screen.getByText("KAN")).toBeInTheDocument();
  });

  it("renders city and name", () => {
    render(<TeamHero team={TEAM} />);
    expect(screen.getByText(/Kansas City/)).toBeInTheDocument();
    expect(screen.getByText("Chiefs")).toBeInTheDocument();
  });

  it("renders context label with conference and division", () => {
    render(<TeamHero team={TEAM} context="AWAY" />);
    expect(screen.getByText(/AWAY/)).toBeInTheDocument();
    expect(screen.getByText(/AFC W/)).toBeInTheDocument();
  });

  it("does not render context row when no context provided", () => {
    const { container } = render(<TeamHero team={TEAM} />);
    // Context row would contain "·"; when absent, no context text
    expect(container.textContent).not.toContain("AWAY");
    expect(container.textContent).not.toContain("HOME");
  });

  it("renders record when provided", () => {
    render(<TeamHero team={TEAM} record="9-2" />);
    expect(screen.getByText(/9-2/)).toBeInTheDocument();
  });

  it("renders rating when provided", () => {
    render(<TeamHero team={TEAM} rating={92.4} />);
    expect(screen.getByText(/92.4/)).toBeInTheDocument();
  });

  it("renders ATS record when provided", () => {
    render(<TeamHero team={TEAM} record="9-2" atsRecord="7-4" />);
    expect(screen.getByText(/ATS 7-4/)).toBeInTheDocument();
  });

  it("falls back to grey mark when primary_color missing", () => {
    const teamNoColor = { ...TEAM, primary_color: null };
    render(<TeamHero team={teamNoColor} />);
    expect(screen.getByText("KAN")).toBeInTheDocument();
  });

  it("respects size prop", () => {
    render(<TeamHero team={TEAM} size={80} />);
    const mark = screen.getByText("KAN");
    expect(mark).toHaveStyle({ width: "80px", height: "80px" });
  });

  it("orientation left by default", () => {
    const { container } = render(<TeamHero team={TEAM} />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveStyle({ justifyContent: "flex-start" });
  });

  it("orientation right flips layout", () => {
    const { container } = render(<TeamHero team={TEAM} orientation="right" />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveStyle({ justifyContent: "flex-end" });
  });
});
