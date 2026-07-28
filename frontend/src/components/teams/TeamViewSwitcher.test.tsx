import {
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  beforeEach,
  describe,
  expect,
  it,
} from "vitest";
import { TestWrapper } from "../../test/testWrapper";
import { TeamViewSwitcher } from "./TeamViewSwitcher";

function renderSwitcher(
  active: "rankings" | "projections",
) {
  return render(
    <TestWrapper>
      <TeamViewSwitcher active={active} />
    </TestWrapper>,
  );
}

describe("TeamViewSwitcher", () => {
  beforeEach(() => {
    window.location.hash = "";
    sessionStorage.clear();
  });

  it("marks Team Rankings as the current page", () => {
    renderSwitcher("rankings");

    expect(
      screen.getByRole("button", {
        name: "Team Rankings",
      }),
    ).toHaveAttribute("aria-current", "page");

    expect(
      screen.getByRole("button", {
        name: "Playoff Projections",
      }),
    ).not.toHaveAttribute("aria-current");
  });

  it("marks Playoff Projections as the current page", () => {
    renderSwitcher("projections");

    expect(
      screen.getByRole("button", {
        name: "Playoff Projections",
      }),
    ).toHaveAttribute("aria-current", "page");
  });

  it("navigates from rankings to projections", async () => {
    const user = userEvent.setup();
    renderSwitcher("rankings");

    await user.click(
      screen.getByRole("button", {
        name: "Playoff Projections",
      }),
    );

    expect(window.location.hash).toBe("#/projections");
  });

  it("navigates from projections to rankings", async () => {
    const user = userEvent.setup();
    renderSwitcher("projections");

    await user.click(
      screen.getByRole("button", {
        name: "Team Rankings",
      }),
    );

    expect(window.location.hash).toBe("#/teams");
  });

  it("does not navigate when the active view is selected", async () => {
    const user = userEvent.setup();
    renderSwitcher("rankings");

    await user.click(
      screen.getByRole("button", {
        name: "Team Rankings",
      }),
    );

    expect(window.location.hash).not.toBe("#/teams");
  });
});
