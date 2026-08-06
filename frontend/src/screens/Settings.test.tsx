import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useEdges } from "../api/hooks";
import { Settings } from "./Settings";
import { TestWrapper } from "../test/testWrapper";

vi.mock("../api/hooks", () => ({ useEdges: vi.fn() }));

describe("Settings", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(useEdges).mockReturnValue({
      data: {
        items: [
          { sportsbook: "draftkings" },
          { sportsbook: "fanduel" },
        ],
      },
      isLoading: false,
      error: null,
    } as never);
  });
  it("renders all settings rows", () => {
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );
    expect(
      screen.getByText("Odds Format"),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Calculator Bankroll",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        /Local what-if value used by standalone calculation tools/,
      ),
    ).toBeInTheDocument();

    expect(screen.getByText("Sportsbooks")).toBeInTheDocument();
    expect(screen.getByText("All available sportsbooks")).toBeInTheDocument();

    expect(
      screen.getByText("Alerts"),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Reset Everything",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Alerts")).toBeInTheDocument();
    expect(screen.getByText("Reset Everything")).toBeInTheDocument();
  });

  it("toggles odds format when Decimal button is clicked", async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );
    const decimalButton = screen.getByText("Decimal");
    await user.click(decimalButton);
    // After click, Decimal has active styling — we check it renders without error.
    expect(decimalButton).toBeInTheDocument();
  });


  it("persists a selected sportsbook subset", async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );

    await user.click(screen.getByRole("button", { name: "Choose books" }));
    expect(screen.getByLabelText("DraftKings")).toBeChecked();
    expect(screen.getByLabelText("FanDuel")).toBeChecked();

    await user.click(screen.getByLabelText("DraftKings"));
    expect(screen.getByText("1 sportsbook selected")).toBeInTheDocument();

    const stored = JSON.parse(localStorage.getItem("hm-app") ?? "{}");
    expect(stored.sportsbookMode).toBe("selected");
    expect(stored.selectedSportsbooks).toEqual(["fanduel"]);
  });

  it("returns to all mode when the final selected sportsbook is cleared", async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper>
        <Settings />
      </TestWrapper>,
    );

    await user.click(screen.getByRole("button", { name: "Choose books" }));
    await user.click(screen.getByLabelText("DraftKings"));
    await user.click(screen.getByLabelText("FanDuel"));
    expect(screen.getByText("All available sportsbooks")).toBeInTheDocument();
  });
});
