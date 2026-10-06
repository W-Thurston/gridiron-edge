import { useState } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CompareChart } from "./CompareChart";
import { segments, summarize, tooltipRows } from "./compareMath";
import type { CompareData, CompareMetric, CompareModelId, CompareSource } from "./compareData";

const BRIER: CompareMetric = {
  key: "brier", label: "Brier", short: "Brier", better: "lower",
  format: (v) => v.toFixed(3), formatDelta: (v) => v.toFixed(3),
};
const ACCURACY: CompareMetric = {
  key: "accuracy", label: "Accuracy", short: "Accuracy", better: "higher",
  format: (v) => `${(v * 100).toFixed(1)}%`, formatDelta: (v) => `${(v * 100).toFixed(1)} pts`,
};

const DATA: CompareData = {
  models: [
    { id: "champion", label: "Champion", source: "logistic", group: "ours" },
    { id: "challenger", label: "Challenger", source: "xgboost", group: "ours" },
    { id: "closing_line", label: "Closing line", source: "closing spread", group: "market" },
  ],
  periods: ["W1", "W2", "W3", "W4"],
  values: {
    brier: {
      champion: [0.24, 0.22, null, 0.21],
      challenger: [0.22, 0.23, 0.25, 0.2],
      closing_line: [null, 0.21, 0.23, 0.22],
    },
    accuracy: {
      champion: [0.6, 0.62, 0.7, 0.64],
      closing_line: [0.65, 0.61, 0.66, 0.6],
    },
  },
};

const SOURCE: CompareSource = { status: "available", data: DATA, metrics: [BRIER, ACCURACY] };

function Harness({ initial = ["champion", "challenger", "closing_line"] as CompareModelId[] }) {
  const [selected, setSelected] = useState<CompareModelId[]>(initial);
  const [metric, setMetric] = useState<string | undefined>(undefined);
  return (
    <CompareChart
      eyebrow="Compare" title="Champion against other models" source={SOURCE}
      selected={selected} onSelectedChange={setSelected} metricKey={metric} onMetricChange={setMetric}
    />
  );
}

function lines(id: string) {
  return document.querySelectorAll(`[data-compare-line="${id}"]`);
}

function hoverPeriod(index: number) {
  const plot = screen.getByRole("group", { name: /by period/ });
  for (let i = 0; i <= index; i += 1) fireEvent.keyDown(plot, { key: "ArrowRight" });
  return screen.getByRole("tooltip");
}

describe("CompareChart", () => {
  it("draws one line per selected model and toggles a model off and on", () => {
    render(<Harness />);
    expect(lines("challenger").length).toBeGreaterThan(0);
    const chip = screen.getByRole("button", { name: "Challenger" });
    fireEvent.click(chip);
    expect(chip).toHaveAttribute("aria-pressed", "false");
    expect(lines("challenger")).toHaveLength(0);
    expect(document.querySelector('[data-compare-row="challenger"]')).toBeNull();
    fireEvent.click(chip);
    expect(chip).toHaveAttribute("aria-pressed", "true");
    expect(lines("challenger").length).toBeGreaterThan(0);
  });

  it("groups the chips into ours and market", () => {
    render(<Harness />);
    const ours = screen.getByRole("group", { name: "Ours models" });
    expect(within(ours).getAllByRole("button").map((b) => b.getAttribute("data-compare-model"))).toEqual(["champion", "challenger"]);
    expect(within(screen.getByRole("group", { name: "Market models" })).getByRole("button", { name: "Closing line" })).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "External models" })).toBeNull();
  });

  it("disables a model with no data for the metric and draws nothing for it", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Accuracy" }));
    const chip = screen.getByRole("button", { name: /Challenger/ });
    expect(chip).toBeDisabled();
    expect(chip).toHaveAttribute("aria-pressed", "false");
    expect(within(chip).getByText("n/a")).toBeInTheDocument();
    expect(chip).toHaveAttribute("title", "Challenger has no Accuracy for this view");
    expect(lines("challenger")).toHaveLength(0);
    expect(screen.getByText("↑ higher is better")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Brier" }));
    expect(screen.getByRole("button", { name: "Challenger" })).not.toBeDisabled();
    expect(screen.getByText("↑ lower is better")).toBeInTheDocument();
  });

  it("lists every visible model in the tooltip, best first", () => {
    render(<Harness />);
    const first = hoverPeriod(0);
    expect(within(first).getByText("W1 · Brier")).toBeInTheDocument();
    // Closing line has no W1 value, so it is left out rather than shown empty.
    expect([...first.querySelectorAll("[data-tooltip-model]")].map((row) => row.getAttribute("data-tooltip-model")))
      .toEqual(["challenger", "champion"]);

    const second = hoverPeriod(0);
    expect(within(second).getByText("W2 · Brier")).toBeInTheDocument();
    expect([...second.querySelectorAll("[data-tooltip-model]")].map((row) => row.getAttribute("data-tooltip-model")))
      .toEqual(["closing_line", "champion", "challenger"]);
  });

  it("sorts a higher-is-better metric with the highest first", () => {
    expect(tooltipRows(DATA, ACCURACY, DATA.models, 2).map((row) => [row.model.id, row.value]))
      .toEqual([["champion", 0.7], ["closing_line", 0.66]]);
    expect(tooltipRows(DATA, BRIER, DATA.models, 2).map((row) => row.model.id))
      .toEqual(["closing_line", "challenger"]);
  });

  it("breaks a line where a model has no value", () => {
    expect(segments([1, 2, null, 4])).toEqual([[[0, 1], [1, 2]], [[3, 4]]]);
    render(<Harness />);
    expect(lines("champion")).toHaveLength(2);
    // The lone W4 point has no neighbour, so it is marked with a dot.
    expect(document.querySelectorAll(".pc-dot")).toHaveLength(1);
  });

  it("summarizes each model against the champion over shared periods", () => {
    const rows = summarize(DATA, BRIER, DATA.models);
    expect(rows.map((row) => row.model.id)).toEqual(["closing_line", "champion", "challenger"]);
    const challenger = rows.find((row) => row.model.id === "challenger")!;
    // Shared periods W1, W2, W4: (-0.02 + 0.01 - 0.01) / 3.
    expect(challenger.delta).toBeCloseTo(-0.00667, 4);
    expect(challenger.best).toBe(2);

    render(<Harness />);
    const row = document.querySelector('[data-compare-row="challenger"]') as HTMLElement;
    expect(within(row).getByText("−0.007 better")).toBeInTheDocument();
    expect(within(row).getByText("best in 2 of 4")).toBeInTheDocument();
    expect(within(row).getByText("Full")).toBeInTheDocument();
    const champion = document.querySelector('[data-compare-row="champion"]') as HTMLElement;
    expect(within(champion).getByText("reference")).toBeInTheDocument();
    expect(within(champion).getByText("W1–W4 · 3 of 4")).toHaveClass("pc-partial");
    const closing = document.querySelector('[data-compare-row="closing_line"]') as HTMLElement;
    expect(within(closing).getByText("W2–W4")).toHaveClass("pc-partial");
    // Shared periods W2 and W4 differ by -0.01 and +0.01.
    expect(within(closing).getByText("Even")).toBeInTheDocument();
  });

  it("asks for a selection when every model is off", () => {
    render(<Harness initial={[]} />);
    expect(screen.getByText("Select at least one model to compare.")).toBeInTheDocument();
    expect(document.querySelector("svg")).toBeNull();
  });

  it("states why there is no comparison instead of drawing one", () => {
    render(
      <CompareChart
        eyebrow="Compare" title="Backtest" selected={[]} onSelectedChange={() => {}} metricKey={undefined} onMetricChange={() => {}}
        source={{ status: "unavailable", reason: "Not persisted", detail: "No per-season rollup." }}
      />,
    );
    expect(screen.getByText("Not persisted")).toBeInTheDocument();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
});
