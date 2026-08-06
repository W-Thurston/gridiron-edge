import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NavProvider, useNav } from "./NavContext";

function Harness() {
  const { route, navigate } = useNav();
  return (
    <div>
      <div data-testid="route">{route.path}</div>
      <div data-testid="params">{JSON.stringify(route.params)}</div>
      <button type="button" onClick={() => navigate("/games")}>Games</button>
      <button
        type="button"
        onClick={() => navigate("/games", { gameId: "2026_01_NE_SEA" })}
      >
        Game detail
      </button>
      <button type="button" onClick={() => navigate("/settings")}>Settings</button>
    </div>
  );
}

function renderNav() {
  return render(
    <NavProvider>
      <Harness />
    </NavProvider>,
  );
}

function traverseTo(hash: string) {
  act(() => {
    window.history.replaceState(null, "", hash);
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
}

describe("NavProvider browser history", () => {
  beforeEach(() => {
    sessionStorage.clear();
    window.history.replaceState(null, "", "#/today");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("pushes normal navigation into browser history", async () => {
    const user = userEvent.setup();
    const pushState = vi.spyOn(window.history, "pushState");
    renderNav();

    await user.click(screen.getByRole("button", { name: "Games" }));

    expect(pushState).toHaveBeenCalledWith(
      { route: { path: "/games", params: {} } },
      "",
      "#/games",
    );
    expect(window.location.hash).toBe("#/games");
    expect(screen.getByTestId("route")).toHaveTextContent("/games");
  });

  it("restores routes and parameters during browser history traversal", () => {
    window.history.replaceState(null, "", "#/settings");
    renderNav();

    traverseTo("#/games?gameId=2026_01_NE_SEA");
    expect(screen.getByTestId("route")).toHaveTextContent("/games");
    expect(screen.getByTestId("params")).toHaveTextContent(
      '{"gameId":"2026_01_NE_SEA"}',
    );

    traverseTo("#/today");
    expect(screen.getByTestId("route")).toHaveTextContent("/today");
    expect(screen.getByTestId("params")).toHaveTextContent("{}");
  });

  it("initializes a direct detail route from the URL", () => {
    window.history.replaceState(
      null,
      "",
      "#/games?gameId=2026_01_NE_SEA",
    );

    renderNav();

    expect(screen.getByTestId("route")).toHaveTextContent("/games");
    expect(screen.getByTestId("params")).toHaveTextContent(
      '{"gameId":"2026_01_NE_SEA"}',
    );
  });

  it("does not push a duplicate entry for the current route", async () => {
    const user = userEvent.setup();
    const pushState = vi.spyOn(window.history, "pushState");
    window.history.replaceState(null, "", "#/games");
    renderNav();

    await user.click(screen.getByRole("button", { name: "Games" }));

    expect(pushState).not.toHaveBeenCalled();
  });

  it("removes browser navigation listeners on unmount", () => {
    const removeEventListener = vi.spyOn(window, "removeEventListener");
    const { unmount } = renderNav();

    unmount();

    expect(removeEventListener).toHaveBeenCalledWith(
      "popstate",
      expect.any(Function),
    );
    expect(removeEventListener).toHaveBeenCalledWith(
      "hashchange",
      expect.any(Function),
    );
  });
});
