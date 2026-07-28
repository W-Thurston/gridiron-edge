import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useProjections } from "../api/hooks";
import { useTeamMetadata } from "../api/team_metadata_hook";
import { TestWrapper } from "../test/testWrapper";
import { PlayoffProjections } from "./PlayoffProjections";

vi.mock("../api/hooks", () => ({
  useProjections: vi.fn(),
}));

vi.mock("../api/team_metadata_hook", () => ({
  useTeamMetadata: vi.fn(),
  useTeamByAbbr: vi.fn(() => null),
}));

const mockedUseProjections = vi.mocked(useProjections);
const mockedUseTeamMetadata = vi.mocked(useTeamMetadata);

const projectionsData = {
  _meta: {
    field_status: {
      "items.clinched": "pending",
      "items.eliminated": "pending",
      "items.elo_delta": {
        status: "blocked",
        blocker: "no_prior_snapshot",
        roadmap: "data",
      },
    },
  },
  items: [
    {
      abbr: "SEA",
      name: "Seattle Seahawks",
      avg_wins: 10.9,
      make_playoffs: 0.78,
      reach_div: 0.55,
      reach_conf: 0.33,
      reach_sb: 0.19,
      win_sb: 0.1,
      elo_delta: null,
      clinched: null,
      eliminated: null,
    },
    {
      abbr: "BUF",
      name: "Buffalo Bills",
      avg_wins: 10.3,
      make_playoffs: 0.73,
      reach_div: 0.49,
      reach_conf: 0.27,
      reach_sb: 0.16,
      win_sb: 0.09,
      elo_delta: null,
      clinched: null,
      eliminated: null,
    },
    {
      abbr: "PHI",
      name: "Philadelphia Eagles",
      avg_wins: 10.7,
      make_playoffs: 0.79,
      reach_div: 0.51,
      reach_conf: 0.29,
      reach_sb: 0.16,
      win_sb: 0.08,
      elo_delta: null,
      clinched: null,
      eliminated: null,
    },
  ],
  total: 3,
  season: "2026-2027",
  computed_at: "2026-07-11T16:28:24+00:00",
  n_simulations: 10000,
};

const teamMetadataData = {
  _meta: {
    field_status: {},
  },
  season: "2026-2027",
  as_of_week: 1,
  items: [
    {
      abbr: "SEA",
      name: "Seahawks",
      city: "Seattle",
      long_name: "Seattle Seahawks",
      conference: "NFC",
      division: "W",
      primary_color: "#002244",
      secondary_color: "#69BE28",
      rating: 1621.45,
      rank: 1,
      record: {
        wins: 0,
        losses: 0,
        ties: 0,
      },
    },
    {
      abbr: "BUF",
      name: "Bills",
      city: "Buffalo",
      long_name: "Buffalo Bills",
      conference: "AFC",
      division: "E",
      primary_color: "#00338D",
      secondary_color: "#C60C30",
      rating: 1602.51,
      rank: 2,
      record: {
        wins: 0,
        losses: 0,
        ties: 0,
      },
    },
    {
      abbr: "PHI",
      name: "Eagles",
      city: "Philadelphia",
      long_name: "Philadelphia Eagles",
      conference: "NFC",
      division: "E",
      primary_color: "#004C54",
      secondary_color: "#A5ACAF",
      rating: 1592.88,
      rank: 3,
      record: {
        wins: 0,
        losses: 0,
        ties: 0,
      },
    },
  ],
  total: 3,
};

function mockLoadedData() {
  mockedUseProjections.mockReturnValue({
    data: projectionsData,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useProjections>);

  mockedUseTeamMetadata.mockReturnValue({
    data: teamMetadataData,
    isLoading: false,
    error: null,
  } as unknown as ReturnType<typeof useTeamMetadata>);
}

function renderScreen() {
  return render(
    <TestWrapper>
      <PlayoffProjections />
    </TestWrapper>,
  );
}

function tableTeamNames(): string[] {
  const rows = screen.getAllByRole("row").slice(1);

  return rows.map((row) => {
    const button = within(row).getByRole("button", {
      name: /Seattle Seahawks|Buffalo Bills|Philadelphia Eagles/,
    });
    return button.textContent ?? "";
  });
}

describe("PlayoffProjections", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.location.hash = "";
    sessionStorage.clear();
    mockLoadedData();
  });

  it("renders the heading and simulation metadata", () => {
    renderScreen();

    expect(
      screen.getByText("Playoff Projections"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("2026-2027 · As of Week 1"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("10,000 simulations"),
    ).toBeInTheDocument();

    expect(screen.getByText(/^Computed /)).toBeInTheDocument();
  });

  it("defaults to Win SB descending", () => {
    renderScreen();

    expect(tableTeamNames()).toEqual([
        "Seattle Seahawks",
        "Buffalo Bills",
        "Philadelphia Eagles",
    ]);

    const winSuperBowlHeader = screen
        .getByRole("button", { name: "Sort by Win SB" })
        .closest("th");

    expect(winSuperBowlHeader).toHaveAttribute(
        "aria-sort",
        "descending",
    );
    });

  it("sorts team names ascending on first selection", async () => {
    const user = userEvent.setup();
    renderScreen();

    await user.click(
      screen.getByRole("button", { name: "Sort by Team" }),
    );

    expect(tableTeamNames()).toEqual([
      "Buffalo Bills",
      "Philadelphia Eagles",
      "Seattle Seahawks",
    ]);
  });

  it("toggles the active sort direction", async () => {
    const user = userEvent.setup();
    renderScreen();

    const winSuperBowlSort = screen.getByRole("button", {
        name: "Sort by Win SB",
    });

    await user.click(winSuperBowlSort);

    expect(tableTeamNames()).toEqual([
        "Philadelphia Eagles",
        "Buffalo Bills",
        "Seattle Seahawks",
    ]);

    const winSuperBowlHeader = screen
        .getByRole("button", { name: "Sort by Win SB" })
        .closest("th");

    expect(winSuperBowlHeader).toHaveAttribute(
        "aria-sort",
        "ascending",
    );
    });

  it("filters by conference while preserving the active sort", async () => {
    const user = userEvent.setup();
    renderScreen();

    const conferenceSelect =
      screen.getByLabelText("Conference");
    const divisionSelect =
      screen.getByLabelText("Division");

    expect(conferenceSelect).toHaveValue("ALL");
    expect(divisionSelect).toBeDisabled();

    await user.selectOptions(conferenceSelect, "AFC");

    expect(conferenceSelect).toHaveValue("AFC");
    expect(divisionSelect).toBeEnabled();
    expect(divisionSelect).toHaveValue("ALL");

    expect(
      screen.getByRole("button", { name: "Buffalo Bills" }),
    ).toBeInTheDocument();

    expect(
      screen.queryByRole("button", {
        name: "Seattle Seahawks",
      }),
    ).not.toBeInTheDocument();

    expect(
      screen.queryByRole("button", {
        name: "Philadelphia Eagles",
      }),
    ).not.toBeInTheDocument();

    expect(screen.getByText("1 of 3 teams")).toBeInTheDocument();

    expect(
      screen
        .getByRole("button", { name: "Sort by Win SB" })
        .closest("th"),
    ).toHaveAttribute("aria-sort", "descending");

    await user.selectOptions(conferenceSelect, "NFC");

    expect(
      screen.getByRole("button", {
        name: "Seattle Seahawks",
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: "Philadelphia Eagles",
      }),
    ).toBeInTheDocument();

    expect(
      screen.queryByRole("button", {
        name: "Buffalo Bills",
      }),
    ).not.toBeInTheDocument();

    expect(screen.getByText("2 of 3 teams")).toBeInTheDocument();
  });

  it("renders team context, current records, and full-cell probability values", () => {
    renderScreen();

    expect(screen.getByText("NFC West")).toBeInTheDocument();
    expect(screen.getByText("AFC East")).toBeInTheDocument();

    expect(screen.getByText("1621")).toBeInTheDocument();
    expect(screen.getByText("1603")).toBeInTheDocument();
    expect(screen.getByText("1593")).toBeInTheDocument();

    expect(screen.getAllByText("0-0")).toHaveLength(3);

    expect(
      screen.getByRole("cell", {
        name: "Seattle Seahawks make playoffs: 78.0%",
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("cell", {
        name: "Seattle Seahawks win Super Bowl: 10.0%",
      }),
    ).toBeInTheDocument();
  });

  it("enables and resets the division filter with conference selection", async () => {
    const user = userEvent.setup();
    renderScreen();

    const conferenceSelect =
      screen.getByLabelText("Conference");
    const divisionSelect =
      screen.getByLabelText("Division");

    expect(divisionSelect).toBeDisabled();

    await user.selectOptions(conferenceSelect, "NFC");
    expect(divisionSelect).toBeEnabled();

    await user.selectOptions(divisionSelect, "E");

    expect(
      screen.getByRole("button", {
        name: "Philadelphia Eagles",
      }),
    ).toBeInTheDocument();

    expect(
      screen.queryByRole("button", {
        name: "Seattle Seahawks",
      }),
    ).not.toBeInTheDocument();

    expect(screen.getByText("1 of 3 teams")).toBeInTheDocument();

    await user.selectOptions(conferenceSelect, "AFC");

    expect(divisionSelect).toHaveValue("ALL");

    expect(
      screen.getByRole("button", {
        name: "Buffalo Bills",
      }),
    ).toBeInTheDocument();

    await user.selectOptions(conferenceSelect, "ALL");

    expect(divisionSelect).toHaveValue("ALL");
    expect(divisionSelect).toBeDisabled();
    expect(screen.getByText("3 of 3 teams")).toBeInTheDocument();
  });

  it("renders a quiet Week 1 Elo state with one explanatory caveat", () => {
    renderScreen();

    expect(
      screen.getAllByLabelText(
        "Elo delta not applicable in Week 1",
      ),
    ).toHaveLength(3);

    expect(
      screen.queryByTitle(
        "Not available: no_prior_snapshot (data)",
      ),
    ).not.toBeInTheDocument();

    expect(
      screen.getByText(
        "1-week Elo change begins after Week 1.",
      ),
    ).toBeInTheDocument();
  });

  it("retains unavailable warnings for missing Elo deltas after Week 1", () => {
    mockedUseTeamMetadata.mockReturnValue({
      data: {
        ...teamMetadataData,
        as_of_week: 2,
      },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useTeamMetadata>);

    renderScreen();

    expect(
      screen.getAllByTitle(
        "Not available: no_prior_snapshot (data)",
      ),
    ).toHaveLength(3);

    expect(
      screen.getByText(
        "Elo movement unavailable without a prior snapshot.",
      ),
    ).toBeInTheDocument();
  });

  it("navigates to the selected team profile", async () => {
    const user = userEvent.setup();
    renderScreen();

    await user.click(
      screen.getByRole("button", {
        name: "Seattle Seahawks",
      }),
    );

    expect(window.location.hash).toBe("#/teams?team=SEA");
  });

  it("sorts by current Elo", async () => {
    const user = userEvent.setup();
    renderScreen();

    const eloSort = screen.getByRole("button", {
      name: "Sort by Elo",
    });

    await user.click(eloSort);

    expect(tableTeamNames()).toEqual([
      "Seattle Seahawks",
      "Buffalo Bills",
      "Philadelphia Eagles",
    ]);

    expect(
      eloSort.closest("th"),
    ).toHaveAttribute("aria-sort", "descending");

    await user.click(eloSort);

    expect(tableTeamNames()).toEqual([
      "Philadelphia Eagles",
      "Buffalo Bills",
      "Seattle Seahawks",
    ]);

    expect(
      eloSort.closest("th"),
    ).toHaveAttribute("aria-sort", "ascending");
  });

  it("keeps rows with missing metadata visible under All", () => {
    mockedUseTeamMetadata.mockReturnValue({
      data: {
        ...teamMetadataData,
        items: teamMetadataData.items.filter(
          (team) => team.abbr !== "SEA",
        ),
      },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useTeamMetadata>);

    renderScreen();

    expect(
      screen.getByRole("button", { name: "Seattle Seahawks" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Metadata unavailable"),
    ).toBeInTheDocument();
  });

  it("renders the loading state", () => {
    mockedUseProjections.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjections>);

    renderScreen();

    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders the actionable empty state", () => {
    mockedUseProjections.mockReturnValue({
      data: {
        ...projectionsData,
        items: [],
        total: 0,
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjections>);

    renderScreen();

    expect(
      screen.getByText(/Run `gridiron sim run` to populate/),
    ).toBeInTheDocument();
  });

  it("renders projections-specific error copy", () => {
    mockedUseProjections.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Unavailable"),
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjections>);

    renderScreen();

    expect(
      screen.getByText("Couldn't load projections"),
    ).toBeInTheDocument();
  });
  it("sorts populated Elo deltas while keeping the dedicated column", async () => {
    const user = userEvent.setup();

    mockedUseProjections.mockReturnValue({
      data: {
        ...projectionsData,
        _meta: {
          field_status: {
            "items.clinched": "pending",
            "items.eliminated": "pending",
          },
        },
        items: [
          {
            ...projectionsData.items[0],
            elo_delta: 8,
          },
          {
            ...projectionsData.items[1],
            elo_delta: -4,
          },
          {
            ...projectionsData.items[2],
            elo_delta: 2,
          },
        ],
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjections>);

    mockedUseTeamMetadata.mockReturnValue({
      data: {
        ...teamMetadataData,
        as_of_week: 2,
      },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useTeamMetadata>);

    renderScreen();

    await user.click(
      screen.getByRole("button", {
        name: "Sort by Elo Δ",
      }),
    );

    expect(tableTeamNames()).toEqual([
      "Seattle Seahawks",
      "Philadelphia Eagles",
      "Buffalo Bills",
    ]);

    expect(
      screen.getByLabelText("Elo delta +8"),
    ).toBeInTheDocument();

    expect(
      screen.getByLabelText("Elo delta -4"),
    ).toBeInTheDocument();
  });
});
