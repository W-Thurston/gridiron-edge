import {
  render,
  screen,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { WinProbabilityCell } from "./WinProbabilityCell";

vi.mock("../field-status/usePendingHighlight", () => ({
  usePendingHighlight: () => ({}),
}));

type CellProps = React.ComponentProps<
  typeof WinProbabilityCell
>;

function renderCell(
  props: Partial<CellProps> = {},
) {
  const defaultProps: CellProps = {
    teamName: "Seattle Seahawks",
    week: 1,
    state: "projected",
    opponent: "Buffalo Bills",
    isHome: true,
    gameDate: "2026-09-09",
    gameTime: "20:20:00",
    winProbability: 0.64,
  };

  return render(
    <table>
      <tbody>
        <tr>
          <WinProbabilityCell
            {...defaultProps}
            {...props}
          />
        </tr>
      </tbody>
    </table>,
  );
}

describe("WinProbabilityCell", () => {
  it("renders a projected win probability", () => {
    renderCell();

    expect(screen.getByText("64%")).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /Seattle Seahawks vs\. Buffalo Bills/,
      }),
    ).toHaveAccessibleName(
      "Seattle Seahawks vs. Buffalo Bills, Week 1, Projected, 64.0% chance to win",
    );
  });

  it("uses an away matchup description", () => {
    renderCell({
      isHome: false,
    });

    expect(
      screen.getByRole("button", {
        name: /Seattle Seahawks at Buffalo Bills/,
      }),
    ).toBeInTheDocument();
  });

  it("shows details on pointer hover", async () => {
    const user = userEvent.setup();
    renderCell();

    const cellButton = screen.getByRole(
      "button",
      {
        name: /Seattle Seahawks vs\. Buffalo Bills/,
      },
    );

    expect(
      screen.queryByRole("tooltip"),
    ).not.toBeInTheDocument();

    await user.hover(cellButton);

    const tooltip = screen.getByRole("tooltip");

    expect(
      within(tooltip).getByText(
        "Seattle Seahawks vs. Buffalo Bills",
      ),
    ).toBeInTheDocument();

    expect(
      within(tooltip).getByText(
        "Week 1 · Sep 9, 2026 · 8:20 PM",
      ),
    ).toBeInTheDocument();

    expect(
      within(tooltip).getByText(
        "Projected · 64.0% chance to win",
      ),
    ).toBeInTheDocument();

    expect(tooltip.parentElement).toBe(document.body);

    await user.unhover(cellButton);

    expect(
      screen.queryByRole("tooltip"),
    ).not.toBeInTheDocument();
  });

  it("shows details on keyboard focus", async () => {
    const user = userEvent.setup();
    renderCell();

    await user.tab();

    expect(
      screen.getByRole("button", {
        name: /Seattle Seahawks vs\. Buffalo Bills/,
      }),
    ).toHaveFocus();

    expect(
      screen.getByRole("tooltip"),
    ).toBeInTheDocument();

    await user.tab();

    expect(
      screen.queryByRole("tooltip"),
    ).not.toBeInTheDocument();
  });

  it("portals and clamps a tooltip near the top-right viewport edge", async () => {
    const user = userEvent.setup();

    const rectSpy = vi
      .spyOn(
        HTMLElement.prototype,
        "getBoundingClientRect",
      )
      .mockReturnValue({
        top: 12,
        bottom: 48,
        left: 1180,
        right: 1230,
        width: 50,
        height: 36,
        x: 1180,
        y: 12,
        toJSON: () => ({}),
      });

    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 1280,
    });

    renderCell();

    await user.hover(
      screen.getByRole("button", {
        name: /Seattle Seahawks vs\. Buffalo Bills/,
      }),
    );

    const tooltip = screen.getByRole("tooltip");

    expect(tooltip.parentElement).toBe(document.body);
    expect(tooltip).toHaveAttribute(
      "data-placement",
      "below",
    );
    expect(tooltip).toHaveStyle({
      position: "fixed",
      left: "912px",
      top: "56px",
      width: "360px",
    });

    rectSpy.mockRestore();
  });

  it("renders played result context", () => {
    renderCell({
      state: "played",
      winProbability: 1,
      actualResult: "W",
    });

    expect(screen.getByText("100%")).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /Played, Win/,
      }),
    ).toBeInTheDocument();
  });

  it("renders a confirmed bye distinctly from zero percent", () => {
    renderCell({
      state: "bye",
      opponent: null,
      isHome: null,
      gameDate: null,
      gameTime: null,
      winProbability: null,
    });

    expect(screen.getByText("BYE")).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: "Seattle Seahawks, Week 1: Bye",
      }),
    ).toBeInTheDocument();

    expect(
      screen.queryByText("0%"),
    ).not.toBeInTheDocument();
  });

  it("renders unavailable without metadata as N/A", () => {
    renderCell({
      state: "unavailable",
      winProbability: null,
    });

    expect(screen.getByText("N/A")).toBeInTheDocument();

    expect(
      screen.getByLabelText(
        "Seattle Seahawks, Week 1: not available",
      ),
    ).toBeInTheDocument();
  });

  it("renders blocked unavailable state", () => {
    renderCell({
      state: "unavailable",
      winProbability: null,
      status: {
        status: "blocked",
        blocker: "no_schedule_data",
        roadmap: "data",
      },
    });

    expect(
      screen.getByTitle(
        "Not available: no_schedule_data (data)",
      ),
    ).toBeInTheDocument();
  });

  it("uses red, neutral, and green full-cell treatments", () => {
    const { rerender } = renderCell({
      winProbability: 0,
    });

    expect(
      screen.getByRole("cell"),
    ).toHaveStyle({
      background:
        "color-mix(in oklab, var(--neg) 40%, transparent)",
    });

    rerender(
      <table>
        <tbody>
          <tr>
            <WinProbabilityCell
              teamName="Seattle Seahawks"
              week={1}
              state="projected"
              opponent="Buffalo Bills"
              isHome
              winProbability={0.5}
            />
          </tr>
        </tbody>
      </table>,
    );

    expect(
      screen.getByRole("cell"),
    ).toHaveStyle({
      background: "var(--bg-2)",
    });

    rerender(
      <table>
        <tbody>
          <tr>
            <WinProbabilityCell
              teamName="Seattle Seahawks"
              week={1}
              state="projected"
              opponent="Buffalo Bills"
              isHome
              winProbability={1}
            />
          </tr>
        </tbody>
      </table>,
    );

    expect(
      screen.getByRole("cell"),
    ).toHaveStyle({
      background:
        "color-mix(in oklab, var(--pos) 40%, transparent)",
    });
  });

  it("marks the played/projected boundary", () => {
    renderCell({
        boundary: true,
    });

    expect(
        screen.getByRole("cell"),
    ).toHaveStyle({
        borderLeftWidth: "2px",
        borderLeftStyle: "solid",
    });
    });
});
