import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { StartingQuarterbacksCard } from "./StartingQuarterbacksCard";

type Block = components["schemas"]["StartingQbBlock"];

function renderCard(startingQb: Block) {
  render(
    <StartingQuarterbacksCard
      startingQb={startingQb}
      awayTeam="IND"
      homeTeam="WAS"
    />,
  );
}

describe("StartingQuarterbacksCard", () => {
  it("shows name, source, snapshot time and a difference flag", () => {
    renderCard({
      status: "available",
      away: {
        player_id: "00-1",
        name: "Daniel Jones",
        source: "nflverse_schedule",
        differs_from_carry_forward: false,
      },
      home: {
        player_id: "00-2",
        name: "Marcus Mariota",
        source: "depth_chart_report",
        projection_as_of: "2026-10-03T12:27:18Z",
        differs_from_carry_forward: true,
      },
    });

    expect(screen.getByText("Marcus Mariota")).toBeInTheDocument();
    expect(
      screen.getByText(/Depth chart and injury report, snapshot 2026-10-03T12:27:18Z/),
    ).toBeInTheDocument();
    expect(screen.getByText("nflverse schedule")).toBeInTheDocument();
    expect(
      screen.getAllByText("Differs from the carried-forward starter"),
    ).toHaveLength(1);
  });

  it("states a product without a record instead of hiding the card", () => {
    renderCard({ status: "not_recorded" });

    expect(
      screen.getByText("Starting quarterbacks were not recorded for this forecast"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Starting quarterback unresolved")).toBeNull();
  });

  it("states an unresolved side and an unnamed starter", () => {
    renderCard({
      status: "starter_unresolved",
      away: { player_id: "00-9", source: "carried_forward" },
      home: null,
    });

    expect(
      screen.getByText("Starting quarterback unresolved for at least one team"),
    ).toBeInTheDocument();
    expect(screen.getByText("00-9 (name unavailable)")).toBeInTheDocument();
    expect(screen.getByText("Starting quarterback unresolved")).toBeInTheDocument();
    expect(screen.getByText("Carry-forward comparison unavailable")).toBeInTheDocument();
  });
});
