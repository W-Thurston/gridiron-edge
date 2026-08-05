import { describe, expect, it } from "vitest";
import {
  isWeeklyComponentUsable,
  weeklyComponentStatusMessage,
  type WeeklyComponent,
} from "./weeklyComponentStatus";

const messages: Array<[WeeklyComponent, string, string]> = [
  ["win", "available", "Win prediction available"],
  [
    "win",
    "policy_unavailable",
    "Win prediction unavailable under the weekly policy",
  ],
  ["win", "forecast_missing", "Win forecast missing"],
  ["win", "forecast_ambiguous", "Win forecast selection ambiguous"],
  ["spread", "available", "Spread available"],
  [
    "spread",
    "win_unavailable",
    "Spread unavailable because Win is unavailable",
  ],
  ["spread", "calibration_unavailable", "Spread calibration unavailable"],
  ["total", "available", "Total available"],
  [
    "total",
    "uncertainty_unavailable",
    "Total available; uncertainty unavailable",
  ],
  [
    "total",
    "policy_unavailable",
    "Total unavailable under the weekly policy",
  ],
  ["total", "forecast_missing", "Total forecast missing"],
  ["total", "forecast_ambiguous", "Total forecast selection ambiguous"],
  ["projected_score", "available", "Projected score available"],
  [
    "projected_score",
    "spread_unavailable",
    "Projected score unavailable because Spread is unavailable",
  ],
  [
    "projected_score",
    "total_unavailable",
    "Projected score unavailable because Total is unavailable",
  ],
  [
    "projected_score",
    "spread_and_total_unavailable",
    "Projected score unavailable because Spread and Total are unavailable",
  ],
];

describe("weeklyComponentStatusMessage", () => {
  it.each(messages)("maps %s status %s", (component, status, expected) => {
    expect(weeklyComponentStatusMessage(component, status)).toBe(expected);
  });

  it("preserves an unknown status in fallback copy", () => {
    expect(weeklyComponentStatusMessage("win", "future_status")).toBe(
      "win status: future_status",
    );
  });
});

describe("isWeeklyComponentUsable", () => {
  it("uses available Win, Spread, and projected scores", () => {
    expect(isWeeklyComponentUsable("win", "available")).toBe(true);
    expect(isWeeklyComponentUsable("spread", "available")).toBe(true);
    expect(isWeeklyComponentUsable("projected_score", "available")).toBe(true);
  });

  it("uses a Total whose point estimate exists without uncertainty", () => {
    expect(isWeeklyComponentUsable("total", "uncertainty_unavailable")).toBe(
      true,
    );
  });

  it("rejects unavailable component states", () => {
    expect(isWeeklyComponentUsable("win", "forecast_missing")).toBe(false);
    expect(isWeeklyComponentUsable("spread", "win_unavailable")).toBe(false);
    expect(isWeeklyComponentUsable("total", "forecast_ambiguous")).toBe(false);
    expect(
      isWeeklyComponentUsable("projected_score", "total_unavailable"),
    ).toBe(false);
  });
});
