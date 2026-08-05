import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WeeklyComponentValue } from "./WeeklyComponentValue";

describe("WeeklyComponentValue", () => {
  it("renders a formatted usable value", () => {
    render(
      <WeeklyComponentValue
        label="Win probability"
        status="available"
        usable
        value={0.58}
        format={(value) => `${Math.round(value * 100)}%`}
        statusMessage="Win prediction available"
      />,
    );

    const value = screen.getByText("58%");
    expect(value).toHaveAttribute("data-weekly-status", "available");
    expect(value).toHaveAttribute("title", "Win prediction available");
  });

  it("renders exact unavailable status copy", () => {
    render(
      <WeeklyComponentValue
        label="Win probability"
        status="forecast_missing"
        usable={false}
        value={null}
        format={String}
        statusMessage="Win forecast missing"
      />,
    );

    const value = screen.getByText("Win forecast missing");
    expect(value).toHaveAttribute(
      "data-weekly-value-state",
      "unavailable",
    );
    expect(value).toHaveAccessibleName("Win forecast missing");
  });

  it("keeps a usable Total point estimate without uncertainty", () => {
    render(
      <WeeklyComponentValue
        label="Total"
        status="uncertainty_unavailable"
        usable
        value={47.5}
        format={(value) => value.toFixed(1)}
        statusMessage="Total available; uncertainty unavailable"
      />,
    );

    expect(screen.getByText("47.5")).toHaveAttribute(
      "title",
      "Total available; uncertainty unavailable",
    );
  });

  it("exposes an inconsistent usable status with no value", () => {
    render(
      <WeeklyComponentValue
        label="Spread"
        status="available"
        usable
        value={null}
        format={(value) => value.toFixed(1)}
        statusMessage="Spread available"
      />,
    );

    const value = screen.getByText("Value unavailable");
    expect(value).toHaveAttribute(
      "data-weekly-value-state",
      "inconsistent",
    );
    expect(value).toHaveAccessibleName(
      "Spread has status available but no value.",
    );
  });
});
