import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useProjectionGrid, useProjections } from "../api/hooks";
import { useTeamMetadata } from "../api/team_metadata_hook";
import { TestWrapper } from "../test/testWrapper";
import { PlayoffProjections } from "./PlayoffProjections";

vi.mock("../api/hooks", () => ({
  useProjectionGrid: vi.fn(),
  useProjections: vi.fn(),
}));


vi.mock("../api/team_metadata_hook", () => ({
  useTeamMetadata: vi.fn(),
  useTeamByAbbr: vi.fn(() => null),
}));

const mockedUseProjections = vi.mocked(useProjections);
const mockedUseProjectionGrid =
  vi.mocked(useProjectionGrid);
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
      name: "Seattle Seahawks",
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
      name: "Buffalo Bills",
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
      name: "Philadelphia Eagles",
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

function makeProjectedWeek({
  week,
  opponent,
  isHome,
  winProbability,
}: {
  week: number;
  opponent: string;
  isHome: boolean;
  winProbability: number;
}) {
  return {
    week,
    state: "projected" as const,
    opponent,
    is_home: isHome,
    game_id: `2026_${String(week).padStart(2, "0")}_TEST`,
    game_date: `2026-09-${String(
      Math.min(week + 8, 28),
    ).padStart(2, "0")}`,
    game_time: "13:00:00",
    win_probability: winProbability,
    actual_result: null,
  };
}

function makeByeWeek(week: number) {
  return {
    week,
    state: "bye" as const,
    opponent: null,
    is_home: null,
    game_id: null,
    game_date: null,
    game_time: null,
    win_probability: null,
    actual_result: null,
  };
}

function makeTeamWeeks({
  opponent,
  baseProbability,
  byeWeek,
}: {
  opponent: string;
  baseProbability: number;
  byeWeek: number;
}) {
  return Array.from({ length: 18 }, (_, index) => {
    const week = index + 1;

    if (week === byeWeek) {
      return makeByeWeek(week);
    }

    return makeProjectedWeek({
      week,
      opponent,
      isHome: week % 2 === 1,
      winProbability: Math.min(
        0.95,
        baseProbability + index * 0.005,
      ),
    });
  });
}

const projectionGridData = {
  _meta: {
    field_status: {},
  },
  season: "2026-2027",
  completed_through_week: 0,
  regular_season_weeks: 18,
  items: [
    {
      abbr: "SEA",
      name: "Seattle Seahawks",
      weeks: makeTeamWeeks({
        opponent: "BUF",
        baseProbability: 0.64,
        byeWeek: 7,
      }),
    },
    {
      abbr: "BUF",
      name: "Buffalo Bills",
      weeks: makeTeamWeeks({
        opponent: "SEA",
        baseProbability: 0.36,
        byeWeek: 11,
      }),
    },
    {
      abbr: "PHI",
      name: "Philadelphia Eagles",
      weeks: makeTeamWeeks({
        opponent: "SEA",
        baseProbability: 0.58,
        byeWeek: 5,
      }),
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

  mockedUseProjectionGrid.mockReturnValue({
    data: projectionGridData,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useProjectionGrid>);

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

async function openWeeklyOutcomes(
  user: ReturnType<typeof userEvent.setup>,
) {
  await user.click(
    screen.getByRole("button", {
      name: "Weekly Outcomes",
    }),
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
      screen.getByRole("button", {
        name: "Playoff Projections",
        current: "page",
      }),
    ).toBeInTheDocument();

    expect(
      screen.getAllByText("Playoff Projections"),
    ).toHaveLength(2);

    expect(
      screen.getByText("2026-2027 · As of Week 1"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("10,000 simulations"),
    ).toBeInTheDocument();

    expect(screen.getByText(/^Computed /)).toBeInTheDocument();
  });

  it("defaults to the Playoff Chances view", () => {
    renderScreen();

    expect(
      screen.getByRole("button", {
        name: "Playoff Chances",
      }),
    ).toHaveAttribute("aria-pressed", "true");

    expect(
      screen.getByRole("button", {
        name: "Weekly Outcomes",
      }),
    ).toHaveAttribute("aria-pressed", "false");

    expect(
      screen.getByRole("button", {
        name: "Sort by Win SB",
      }),
    ).toBeInTheDocument();
  });

  it("defaults to Win SB descending", () => {
    renderScreen();

    expect(tableTeamNames()).toEqual([
      "Seattle Seahawks",
      "Buffalo Bills",
      "Philadelphia Eagles",
    ]);

    const winSuperBowlHeader = screen
      .getByRole("button", {
        name: "Sort by Win SB",
      })
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
      .getByRole("button", {
        name: "Sort by Win SB",
      })
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

  it("renders the all-projected preseason weekly grid", async () => {
    const user = userEvent.setup();
    renderScreen();

    await openWeeklyOutcomes(user);

    expect(
      screen.getByRole("button", {
        name: "Weekly Outcomes",
      }),
    ).toHaveAttribute("aria-pressed", "true");

    expect(
      screen.queryByRole("button", {
        name: "Sort by Win SB",
      }),
    ).not.toBeInTheDocument();

    expect(
      screen.queryByText("Played Games"),
    ).not.toBeInTheDocument();

    expect(
      screen.getByText("Projected Games"),
    ).toBeInTheDocument();

    for (let week = 1; week <= 18; week += 1) {
      expect(
        screen.getByRole("columnheader", {
          name: `W${week}`,
        }),
      ).toBeInTheDocument();
    }

    expect(
      screen.getByText(
        "Season has not started — all games are projected.",
      ),
    ).toBeInTheDocument();
  });
  it("renders weekly percentages and confirmed byes distinctly", async () => {
    const user = userEvent.setup();
    renderScreen();

    await openWeeklyOutcomes(user);

    expect(
      screen.getByRole("button", {
        name: /Seattle Seahawks vs\. Buffalo Bills · Week 1 ·/,
      }),
    ).toHaveTextContent("64%");

    expect(
      screen.getByRole("button", {
        name: "Seattle Seahawks, Week 7: Bye",
      }),
    ).toHaveTextContent("BYE");

    const seattleRow = screen
      .getByRole("button", {
        name: "Seattle Seahawks",
      })
      .closest("tr");

    expect(seattleRow).not.toBeNull();

    if (seattleRow) {
      expect(
        within(seattleRow).queryByText("0%"),
      ).not.toBeInTheDocument();
    }
  });

  it("applies conference filtering to Weekly Outcomes", async () => {
    const user = userEvent.setup();
    renderScreen();

    await openWeeklyOutcomes(user);

    await user.selectOptions(
      screen.getByLabelText("Conference"),
      "AFC",
    );

    expect(
      screen.getByRole("button", {
        name: "Buffalo Bills",
      }),
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

    expect(
      screen.getByText("1 of 3 teams"),
    ).toBeInTheDocument();
  });
  it("applies dependent division filtering to Weekly Outcomes", async () => {
    const user = userEvent.setup();
    renderScreen();

    await openWeeklyOutcomes(user);

    const conferenceSelect =
      screen.getByLabelText("Conference");
    const divisionSelect =
      screen.getByLabelText("Division");

    await user.selectOptions(conferenceSelect, "NFC");
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

    expect(
      screen.queryByRole("button", {
        name: "Buffalo Bills",
      }),
    ).not.toBeInTheDocument();
  });
  it("preserves filters while switching projection views", async () => {
    const user = userEvent.setup();
    renderScreen();

    const conferenceSelect =
      screen.getByLabelText("Conference");
    const divisionSelect =
      screen.getByLabelText("Division");

    await user.selectOptions(conferenceSelect, "NFC");
    await user.selectOptions(divisionSelect, "E");

    await openWeeklyOutcomes(user);

    expect(conferenceSelect).toHaveValue("NFC");
    expect(divisionSelect).toHaveValue("E");

    await user.click(
      screen.getByRole("button", {
        name: "Playoff Chances",
      }),
    );

    expect(conferenceSelect).toHaveValue("NFC");
    expect(divisionSelect).toHaveValue("E");

    expect(
      screen.getByRole("button", {
        name: "Philadelphia Eagles",
      }),
    ).toBeInTheDocument();
  });
  it("renders played and projected groups with a boundary", async () => {
    const user = userEvent.setup();

    const mixedWeeks =
      projectionGridData.items[0].weeks.map((week) => {
        if (week.week === 1) {
          return {
            ...week,
            state: "played" as const,
            win_probability: 1,
            actual_result: "W" as const,
          };
        }

        return week;
      });

    mockedUseProjectionGrid.mockReturnValue({
      data: {
        ...projectionGridData,
        completed_through_week: 1,
        items: [
          {
            ...projectionGridData.items[0],
            weeks: mixedWeeks,
          },
        ],
        total: 1,
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjectionGrid>);

    renderScreen();
    await openWeeklyOutcomes(user);

    expect(
      screen.getByText("Played Games"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("Projected Games"),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /Played — win/,
      }),
    ).toBeInTheDocument();

    const weekTwoHeader = screen.getByRole(
      "columnheader",
      { name: "W2" },
    );

    expect(weekTwoHeader).toHaveStyle({
      borderLeftWidth: "2px",
      borderLeftStyle: "solid",
    });

    expect(
      screen.getByText(
        /Played games show fixed outcomes/,
      ),
    ).toBeInTheDocument();
  });
  it("scopes weekly loading state to Weekly Outcomes", async () => {
    const user = userEvent.setup();

    mockedUseProjectionGrid.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjectionGrid>);

    renderScreen();

    expect(
      screen.queryByText("Loading weekly projections…"),
    ).not.toBeInTheDocument();

    await openWeeklyOutcomes(user);

    expect(
      screen.getByText("Loading weekly projections…"),
    ).toBeInTheDocument();
  });
  it("renders weekly-grid-specific error copy", async () => {
    const user = userEvent.setup();

    mockedUseProjectionGrid.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Unavailable"),
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjectionGrid>);

    renderScreen();
    await openWeeklyOutcomes(user);

    expect(
      screen.getByText(
        "Couldn't load weekly projections",
      ),
    ).toBeInTheDocument();
  });
  it("renders the weekly-grid empty state", async () => {
    const user = userEvent.setup();

    mockedUseProjectionGrid.mockReturnValue({
      data: {
        ...projectionGridData,
        items: [],
        total: 0,
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useProjectionGrid>);

    renderScreen();
    await openWeeklyOutcomes(user);

    expect(
      screen.getByText(
        /No weekly projection grid found/,
      ),
    ).toBeInTheDocument();
  });
});
