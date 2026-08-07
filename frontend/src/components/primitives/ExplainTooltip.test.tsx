import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ExplainTooltip } from "./ExplainTooltip";

function renderTooltip() {
  return render(
    <ExplainTooltip
      accessibleLabel="Explain Arizona plus 10.5 at minus 110"
      title="Arizona +10.5 at -110"
      sections={[
        { label: "Bet outcome", text: "Arizona can lose by 10 points or fewer." },
        { label: "Price", text: "A $110 stake produces $100 profit." },
      ]}
    >
      <span>+10.5 -110</span>
    </ExplainTooltip>,
  );
}

describe("ExplainTooltip", () => {
  it("opens on hover and portals the explanation", async () => {
    const user = userEvent.setup();
    renderTooltip();
    const trigger = screen.getByRole("button", { name: /explain arizona/i });

    await user.hover(trigger);

    const tooltip = screen.getByRole("tooltip");
    expect(tooltip.parentElement).toBe(document.body);
    expect(within(tooltip).getByText("Bet outcome")).toBeInTheDocument();
    expect(within(tooltip).getByText(/\$110 stake/)).toBeInTheDocument();
  });

  it("opens on keyboard focus", async () => {
    const user = userEvent.setup();
    renderTooltip();

    await user.tab();

    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");
  });

  it("pins and unpins on click for touch interaction", async () => {
    const user = userEvent.setup();
    renderTooltip();
    const trigger = screen.getByRole("button");

    await user.click(trigger);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    await user.click(trigger);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
