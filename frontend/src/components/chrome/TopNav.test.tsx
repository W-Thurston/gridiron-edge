import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TopNav } from "./TopNav";
import { useNav } from "../../context/NavContext";

const navigate = vi.fn();
let currentPath = "/today";

vi.mock("../../context/NavContext", () => ({
  useNav: vi.fn(),
}));

vi.mock("../../context/AppStateContext", () => ({
  useAppState: () => ({
    state: {
      alerts: 0,
    },
  }),
}));

vi.mock("../../context/BetSlipContext", () => ({
  useBetSlip: () => ({
    legs: [],
  }),
}));

vi.mock("./icons", () => ({
  BellIcon: () => <span />,
  BetSlipIcon: () => <span />,
  SearchIcon: () => <span />,
}));

describe("TopNav", () => {
  beforeEach(() => {
    navigate.mockReset();
    currentPath = "/today";
    vi.mocked(useNav).mockImplementation(() => ({
      route: {
        path: currentPath,
        params: {},
      },
      navigate,
    }));
  });

  it.each([
    ["/teams", "Teams"],
    ["/projections", "Teams"],
    ["/games", "Games"],
  ])("marks %s under the %s navigation family", (path, label) => {
    currentPath = path;

    render(<TopNav />);

    expect(screen.getByRole("button", { name: label })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("does not mark Teams active for Games", () => {
    currentPath = "/games";

    render(<TopNav />);

    expect(screen.getByRole("button", { name: "Teams" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("navigates to the Teams rankings route", () => {
    currentPath = "/projections";

    render(<TopNav />);
    fireEvent.click(screen.getByRole("button", { name: "Teams" }));

    expect(navigate).toHaveBeenCalledWith("/teams");
  });
});
