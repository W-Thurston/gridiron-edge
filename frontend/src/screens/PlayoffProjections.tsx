import { useMemo, useState } from "react";
import { useProjectionGrid, useProjections } from "../api/hooks";
import { useTeamMetadata } from "../api/team_metadata_hook";
import { ErrorCard } from "../components/error/ErrorCard";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { StatusPill } from "../components/projections/StatusPill";
import { HeatCell } from "../components/primitives/HeatCell";
import { WinProbabilityCell } from "../components/primitives/WinProbabilityCell";
import { TeamViewSwitcher } from "../components/teams/TeamViewSwitcher";
import {
  SortableHeader,
  type SortDirection,
} from "../components/primitives/SortableHeader";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";

type ConferenceFilter = "ALL" | "AFC" | "NFC";
type DivisionFilter = "ALL" | "N" | "S" | "E" | "W";
type ProjectionView = "chances" | "weekly";

type SortKey =
  | "team"
  | "elo"
  | "elo_delta"
  | "avg_wins"
  | "make_playoffs"
  | "reach_div"
  | "reach_conf"
  | "reach_sb"
  | "win_sb";

type ProjectionItem = NonNullable<
  NonNullable<ReturnType<typeof useProjections>["data"]>["items"]
>[number];

type ProjectionGridData = NonNullable<
  ReturnType<typeof useProjectionGrid>["data"]
>;

type ProjectionGridTeam = NonNullable<
  ProjectionGridData["items"]
>[number];

type TeamMetadataItem = NonNullable<
  NonNullable<ReturnType<typeof useTeamMetadata>["data"]>["items"]
>[number];

type EnrichedProjection = {
  projection: ProjectionItem;
  metadata: TeamMetadataItem | undefined;
};

type EnrichedWeeklyProjection = {
  projection: ProjectionGridTeam;
  metadata: TeamMetadataItem | undefined;
};

export function PlayoffProjections() {
  const { data, isLoading, error, refetch } = useProjections();
  const {
    data: gridData,
    isLoading: isGridLoading,
    error: gridError,
    refetch: refetchGrid,
  } = useProjectionGrid();
  const { data: teamMetadata } = useTeamMetadata();
  const { navigate } = useNav();

  const [view, setView] =
    useState<ProjectionView>("chances");
  const [conference, setConference] =
    useState<ConferenceFilter>("ALL");
  const [division, setDivision] =
    useState<DivisionFilter>("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("win_sb");
  const [sortDirection, setSortDirection] =
    useState<SortDirection>("desc");

  const metadataByAbbr = useMemo(
    () =>
      new Map(
        (teamMetadata?.items ?? []).map((team) => [team.abbr, team]),
      ),
    [teamMetadata?.items],
  );

  const visibleRows = useMemo(() => {
    const rows: EnrichedProjection[] = (data?.items ?? []).map(
      (projection) => ({
        projection,
        metadata: metadataByAbbr.get(projection.abbr),
      }),
    );

    const filtered = rows.filter(({ metadata }) =>
      matchesTeamFilters(
        metadata,
        conference,
        division,
      ),
    );

    return filtered.toSorted((a, b) =>
      compareRows(a, b, sortKey, sortDirection),
    );
  }, [
    conference,
    data?.items,
    division,
    metadataByAbbr,
    sortDirection,
    sortKey,
  ]);

  const visibleWeeklyRows = useMemo(() => {
    const rows: EnrichedWeeklyProjection[] = (
      gridData?.items ?? []
    ).map((projection) => ({
      projection,
      metadata: metadataByAbbr.get(projection.abbr),
    }));

    return rows.filter(({ metadata }) =>
      matchesTeamFilters(
        metadata,
        conference,
        division,
      ),
    );
  }, [
    conference,
    division,
    gridData?.items,
    metadataByAbbr,
  ]);

  const handleSort = (nextKey: SortKey) => {
    if (nextKey === sortKey) {
      setSortDirection((current) =>
        current === "asc" ? "desc" : "asc",
      );
      return;
    }

    setSortKey(nextKey);
    setSortDirection(nextKey === "team" ? "asc" : "desc");
  };

  const handleConferenceChange = (
    nextConference: ConferenceFilter,
  ) => {
    setConference(nextConference);
    setDivision("ALL");
  };

  const handleDivisionChange = (
    nextDivision: DivisionFilter,
  ) => {
    setDivision(nextDivision);
  };

  const fieldStatus = data?._meta?.field_status;

  return (
    <div>
      <TeamViewSwitcher active="projections" />

      <div className="hm-card" style={{ padding: 24 }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 24,
            marginBottom: 18,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div
              className="upper dim"
              style={{ fontSize: 10, marginBottom: 5 }}
            >
              Playoff Projections
            </div>
            <div
              className="serif"
              style={{
                color: "var(--ink-2)",
                fontSize: 20,
                fontStyle: "italic",
              }}
            >
              Monte Carlo estimates for each team’s path through the
              postseason.
            </div>
          </div>

          <SimulationMetadata
            season={data?.season}
            asOfWeek={teamMetadata?.as_of_week}
            computedAt={data?.computed_at}
            nSimulations={data?.n_simulations}
            nSimulationsStatus={
              fieldStatus?.n_simulations as FieldStatus | undefined
            }
          />
        </header>
        <nav
          aria-label="Projection views"
          style={{
            display: "flex",
            gap: 4,
            marginBottom: 14,
            padding: 3,
            width: "fit-content",
            border: "1px solid var(--line-soft)",
            borderRadius: 5,
            background: "var(--bg-1)",
          }}
        >
          <ProjectionViewButton
            active={view === "chances"}
            onClick={() => setView("chances")}
          >
            Playoff Chances
          </ProjectionViewButton>

          <ProjectionViewButton
            active={view === "weekly"}
            onClick={() => setView("weekly")}
          >
            Weekly Outcomes
          </ProjectionViewButton>
        </nav>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            marginBottom: 14,
            flexWrap: "wrap",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: 10,
              flexWrap: "wrap",
            }}
          >
            <FilterSelect
              id="projections-conference"
              label="Conference"
              value={conference}
              onChange={(value) =>
                handleConferenceChange(value as ConferenceFilter)
              }
              options={[
                { value: "ALL", label: "All Conferences" },
                { value: "AFC", label: "AFC" },
                { value: "NFC", label: "NFC" },
              ]}
            />

            <FilterSelect
              id="projections-division"
              label="Division"
              value={division}
              disabled={conference === "ALL"}
              onChange={(value) =>
                handleDivisionChange(value as DivisionFilter)
              }
              options={[
                { value: "ALL", label: "All Divisions" },
                { value: "N", label: "North" },
                { value: "S", label: "South" },
                { value: "E", label: "East" },
                { value: "W", label: "West" },
              ]}
            />
          </div>

          {view === "chances" && data && (
            <div className="mono dim" style={{ fontSize: 10 }}>
              {visibleRows.length} of{" "}
              {data.total ?? data.items?.length ?? 0} teams
            </div>
          )}

          {view === "weekly" && gridData && (
            <div className="mono dim" style={{ fontSize: 10 }}>
              {visibleWeeklyRows.length} of{" "}
              {gridData.total ?? gridData.items?.length ?? 0} teams
            </div>
          )}
        </div>
        {view === "chances" && (
          <>
            {isLoading && <div className="dim">Loading…</div>}

            {error && (
              <ErrorCard
                error={error}
                onRetry={() => refetch()}
                title="Couldn't load projections"
              />
            )}

            {data && (data.items ?? []).length === 0 && (
              <div className="dim mono" style={{ fontSize: 12 }}>
                No projections found. Run `gridiron sim run` to populate.
              </div>
            )}

          {data && (data.items ?? []).length > 0 && (
            <>
              <div style={{ overflowX: "auto" }}>
                <table
                  className="mono tnum"
                  style={{
                    width: "100%",
                    minWidth: 1080,
                    fontSize: 12,
                    borderCollapse: "collapse",
                  }}
                >
                  <thead>
                    <tr>
                      <SortableHeader
                        label="Team"
                        active={sortKey === "team"}
                        direction={sortDirection}
                        onClick={() => handleSort("team")}
                      />
                      <SortableHeader
                        label="Elo"
                        active={sortKey === "elo"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("elo")}
                      />

                      <SortableHeader
                        label="Elo Δ"
                        active={sortKey === "elo_delta"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("elo_delta")}
                      />

                      <th
                        scope="col"
                        style={{
                          padding: "8px 12px 8px 0",
                          textAlign: "right",
                          color: "var(--ink-3)",
                          fontWeight: 400,
                        }}
                      >
                        Record
                      </th>
                      <SortableHeader
                        label="Avg Wins"
                        active={sortKey === "avg_wins"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("avg_wins")}
                      />
                      <SortableHeader
                        label="Playoffs"
                        active={sortKey === "make_playoffs"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("make_playoffs")}
                      />
                      <SortableHeader
                        label="Div. Round"
                        active={sortKey === "reach_div"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("reach_div")}
                      />
                      <SortableHeader
                        label="Conf. Champ."
                        active={sortKey === "reach_conf"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("reach_conf")}
                      />
                      <SortableHeader
                        label="Make SB"
                        active={sortKey === "reach_sb"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("reach_sb")}
                      />
                      <SortableHeader
                        label="Win SB"
                        active={sortKey === "win_sb"}
                        direction={sortDirection}
                        align="right"
                        onClick={() => handleSort("win_sb")}
                      />
                    </tr>
                  </thead>

                  <tbody>
                    {visibleRows.map(({ projection, metadata }) => (
                      <tr
                        key={projection.abbr}
                        className="proj-row"
                        style={{
                          borderTop: "1px solid var(--line-soft)",
                        }}
                      >
                        <td style={{ padding: "10px 14px 10px 0" }}>
                          <ProjectionTeamIdentity
                            abbr={projection.abbr}
                            name={projection.name}
                            metadata={metadata}
                            clinched={projection.clinched}
                            eliminated={projection.eliminated}
                            onNavigate={() =>
                              navigate("/teams", {
                                team: projection.abbr,
                              })
                            }
                          />
                        </td>
                        <td
                          style={{
                            padding: "10px 12px 10px 0",
                            textAlign: "right",
                            color: "var(--ink-2)",
                          }}
                        >
                          {metadata?.rating != null
                            ? Math.round(metadata.rating)
                            : "N/A"}
                        </td>

                        <td
                          style={{
                            padding: "10px 12px 10px 0",
                            textAlign: "right",
                          }}
                        >
                          <EloDelta
                            value={projection.elo_delta}
                            status={
                              fieldStatus?.["items.elo_delta"] as
                                | FieldStatus
                                | undefined
                            }
                            asOfWeek={teamMetadata?.as_of_week}
                          />
                        </td>

                        <td
                          style={{
                            padding: "10px 12px 10px 0",
                            textAlign: "right",
                            color: "var(--ink-2)",
                          }}
                        >
                          {formatRecord(metadata?.record)}
                        </td>
                        <td
                          style={{
                            padding: "10px 12px 10px 0",
                            textAlign: "right",
                            color: "var(--ink-2)",
                          }}
                        >
                          {projection.avg_wins?.toFixed(1) ?? "N/A"}
                        </td>

                        <HeatCell
                          value={projection.make_playoffs}
                          label={`${projection.name} make playoffs`}
                          status={
                            fieldStatus?.["items.make_playoffs"] as
                              | FieldStatus
                              | undefined
                          }
                        />

                        <HeatCell
                          value={projection.reach_div}
                          label={`${projection.name} reach divisional round`}
                          status={
                            fieldStatus?.["items.reach_div"] as
                              | FieldStatus
                              | undefined
                          }
                        />

                        <HeatCell
                          value={projection.reach_conf}
                          label={`${projection.name} reach conference championship`}
                          status={
                            fieldStatus?.["items.reach_conf"] as
                              | FieldStatus
                              | undefined
                          }
                        />

                        <HeatCell
                          value={projection.reach_sb}
                          label={`${projection.name} reach Super Bowl`}
                          status={
                            fieldStatus?.["items.reach_sb"] as
                              | FieldStatus
                              | undefined
                          }
                        />

                        <HeatCell
                          value={projection.win_sb}
                          label={`${projection.name} win Super Bowl`}
                          status={
                            fieldStatus?.["items.win_sb"] as
                              | FieldStatus
                              | undefined
                          }
                        />
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <ProjectionLegend
                asOfWeek={teamMetadata?.as_of_week}
                eloStatus={
                  fieldStatus?.["items.elo_delta"] as
                    | FieldStatus
                    | undefined
                }
                clinchedStatus={
                  fieldStatus?.["items.clinched"] as
                    | FieldStatus
                    | undefined
                }
              />
            </>
          )}
        </>
      )}
      {view === "weekly" && (
        <>
          {isGridLoading && (
            <div className="dim">
              Loading weekly projections…
            </div>
          )}

          {gridError && (
            <ErrorCard
              error={gridError}
              onRetry={() => refetchGrid()}
              title="Couldn't load weekly projections"
            />
          )}

          {gridData &&
            (gridData.items ?? []).length === 0 && (
              <div
                className="dim mono"
                style={{ fontSize: 12 }}
              >
                No weekly projection grid found. Run `gridiron sim run`
                to populate.
              </div>
            )}

          {gridData &&
            (gridData.items ?? []).length > 0 && (
              <WeeklyOutcomesGrid
                data={gridData}
                rows={visibleWeeklyRows}
                metadataByAbbr={metadataByAbbr}
                weeklyStatus={
                  gridData._meta?.field_status?.[
                    "items.weeks"
                  ] as FieldStatus | undefined
                }
                onNavigateTeam={(abbr) =>
                  navigate("/teams", { team: abbr })
                }
              />
            )}
        </>
      )}
      </div>
    </div>
  );
}

function ProjectionViewButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      style={{
        padding: "5px 11px",
        border: 0,
        borderRadius: 3,
        backgroundColor: active
          ? "var(--pos)"
          : "transparent",
        color: active
          ? "var(--bg)"
          : "var(--ink-3)",
        fontFamily: "var(--f-sans)",
        fontSize: 11,
        cursor: active ? "default" : "pointer",
      }}
    >
      {children}
    </button>
  );
}

function WeeklyOutcomesGrid({
  data,
  rows,
  metadataByAbbr,
  weeklyStatus,
  onNavigateTeam,
}: {
  data: ProjectionGridData;
  rows: EnrichedWeeklyProjection[];
  metadataByAbbr: Map<string, TeamMetadataItem>;
  weeklyStatus?: FieldStatus;
  onNavigateTeam: (abbr: string) => void;
}) {
  const completedThroughWeek =
    data.completed_through_week;
  const regularSeasonWeeks =
    data.regular_season_weeks;

  const playedWeekCount = Math.min(
    completedThroughWeek,
    regularSeasonWeeks,
  );
  const projectedWeekCount =
    regularSeasonWeeks - playedWeekCount;

  const firstProjectedWeek =
    playedWeekCount > 0 &&
    playedWeekCount < regularSeasonWeeks
      ? playedWeekCount + 1
      : null;

  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table
          className="mono tnum"
          style={{
            width: "100%",
            minWidth:
              220 + regularSeasonWeeks * 54,
            fontSize: 11,
            borderCollapse: "collapse",
          }}
        >
          <thead>
            <tr>
              <th
                scope="col"
                rowSpan={2}
                style={{
                  position: "sticky",
                  left: 0,
                  zIndex: 4,
                  minWidth: 220,
                  padding: "8px 12px",
                  textAlign: "left",
                  background: "var(--bg)",
                  color: "var(--ink-3)",
                  borderRight:
                    "1px solid var(--line)",
                }}
              >
                Team
              </th>

              {playedWeekCount > 0 && (
                <th
                  scope="colgroup"
                  colSpan={playedWeekCount}
                  style={{
                    padding: "7px 8px",
                    textAlign: "center",
                    color: "var(--ink-3)",
                    borderBottom:
                      "1px solid var(--line-soft)",
                  }}
                >
                  Played Games
                </th>
              )}

              {projectedWeekCount > 0 && (
                <th
                  scope="colgroup"
                  colSpan={projectedWeekCount}
                  style={{
                    padding: "7px 8px",
                    textAlign: "center",
                    color: "var(--ink-3)",
                    borderBottom:
                      "1px solid var(--line-soft)",
                    borderLeft:
                      playedWeekCount > 0
                        ? "2px solid var(--line)"
                        : undefined,
                  }}
                >
                  Projected Games
                </th>
              )}
            </tr>

            <tr>
              {Array.from(
                { length: regularSeasonWeeks },
                (_, index) => index + 1,
              ).map((week) => {
                const boundary =
                  week === firstProjectedWeek;

                return (
                  <th
                    key={week}
                    scope="col"
                    style={{
                      minWidth: 54,
                      padding: "6px 4px",
                      textAlign: "center",
                      color: "var(--ink-4)",
                      borderLeftWidth: boundary
                        ? 2
                        : 1,
                      borderLeftStyle: "solid",
                      borderLeftColor: boundary
                        ? "var(--line)"
                        : "var(--line-soft)",
                    }}
                  >
                    W{week}
                  </th>
                );
              })}
            </tr>
          </thead>

          <tbody>
            {rows.map(({ projection, metadata }) => (
              <tr
                key={projection.abbr}
                style={{
                  borderTop:
                    "1px solid var(--line-soft)",
                }}
              >
                <td
                  style={{
                    position: "sticky",
                    left: 0,
                    zIndex: 3,
                    minWidth: 220,
                    padding: "10px 14px 10px 0",
                    background: "var(--bg)",
                    borderRight:
                      "1px solid var(--line)",
                  }}
                >
                  <ProjectionTeamIdentity
                    abbr={projection.abbr}
                    name={projection.name}
                    metadata={metadata}
                    onNavigate={() =>
                      onNavigateTeam(projection.abbr)
                    }
                  />
                </td>

                {projection.weeks.map((week) => {
                  const opponentName =
                    week.opponent == null
                      ? null
                      : metadataByAbbr.get(
                          week.opponent,
                        )?.name ??
                        week.opponent;

                  return (
                    <WinProbabilityCell
                      key={week.week}
                      teamName={projection.name}
                      week={week.week}
                      state={week.state}
                      opponent={opponentName}
                      isHome={week.is_home}
                      gameDate={week.game_date}
                      gameTime={week.game_time}
                      winProbability={
                        week.win_probability
                      }
                      actualResult={
                        week.actual_result
                      }
                      status={weeklyStatus}
                      boundary={
                        week.week ===
                        firstProjectedWeek
                      }
                    />
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <WeeklyOutcomesLegend
        completedThroughWeek={
          completedThroughWeek
        }
      />
    </>
  );
}

function WeeklyOutcomesLegend({
  completedThroughWeek,
}: {
  completedThroughWeek: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 16,
        marginTop: 14,
        paddingTop: 12,
        borderTop:
          "1px solid var(--line-soft)",
        color: "var(--ink-4)",
        fontSize: 10,
        flexWrap: "wrap",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
        }}
      >
        <span>Lower chance to win</span>

        <span
          aria-hidden="true"
          style={{
            width: 18,
            height: 8,
            borderRadius: 2,
            background:
              "color-mix(in oklab, var(--neg) 34%, transparent)",
          }}
        />

        <span
          aria-hidden="true"
          style={{
            width: 18,
            height: 8,
            borderRadius: 2,
            background: "var(--bg-2)",
          }}
        />

        <span
          aria-hidden="true"
          style={{
            width: 18,
            height: 8,
            borderRadius: 2,
            background:
              "color-mix(in oklab, var(--pos) 34%, transparent)",
          }}
        />

        <span>Higher chance to win</span>
      </div>

      {completedThroughWeek === 0 ? (
        <span>
          Season has not started — all games are
          projected.
        </span>
      ) : (
        <span>
          Played games show fixed outcomes from the
          current simulation artifact; projected games
          show simulated chance to win.
        </span>
      )}
    </div>
  );
}

function ProjectionTeamIdentity({
  abbr,
  name,
  metadata,
  clinched,
  eliminated,
  onNavigate,
}: {
  abbr: string;
  name: string;
  metadata: TeamMetadataItem | undefined;
  clinched?: boolean | null;
  eliminated?: boolean | null;
  onNavigate: () => void;
}) {
  const division = formatDivision(
    metadata?.conference,
    metadata?.division,
  );

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 9,
        minWidth: 210,
      }}
    >
      <TeamMark abbr={abbr} />

      <div style={{ minWidth: 0, flex: 1 }}>
        <button
          type="button"
          onClick={onNavigate}
          style={{
            display: "block",
            width: "100%",
            padding: 0,
            border: 0,
            backgroundColor: "transparent",
            color: "var(--ink-2)",
            font: "inherit",
            textAlign: "left",
            cursor: "pointer",
          }}
        >
          {name}
        </button>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            minHeight: 17,
            marginTop: 2,
            color: "var(--ink-4)",
            fontSize: 10,
          }}
        >
          <span>
            {division ?? "Metadata unavailable"}
          </span>

          <StatusPill
            clinched={clinched}
            eliminated={eliminated}
          />
        </div>
      </div>
    </div>
  );
}

function EloDelta({
  value,
  status,
  asOfWeek,
}: {
  value: number | null | undefined;
  status?: FieldStatus;
  asOfWeek: number | null | undefined;
}) {
  if (value == null) {
    if (asOfWeek === 1) {
      return (
        <span
          className="mono tnum"
          title="1-week Elo change begins after Week 1"
          aria-label="Elo delta not applicable in Week 1"
          style={{ color: "var(--ink-4)" }}
        >
          —
        </span>
      );
    }

    if (status === "pending") {
      return <PendingField placeholder="Elo Δ" />;
    }

    if (status) {
      return (
        <BlockedField
          blocker={status.blocker}
          roadmap={status.roadmap}
          placeholder="Elo Δ"
        />
      );
    }

    return (
      <span
        className="mono tnum"
        title="Elo delta not available"
        style={{ color: "var(--ink-4)" }}
      >
        N/A
      </span>
    );
  }

  const color =
    value > 0
      ? "var(--pos)"
      : value < 0
        ? "var(--neg)"
        : "var(--ink-3)";

  const background =
    value > 0
      ? "color-mix(in oklab, var(--pos) 14%, transparent)"
      : value < 0
        ? "color-mix(in oklab, var(--neg) 14%, transparent)"
        : "var(--bg-2)";

  const signed =
    value > 0 ? `+${value.toFixed(0)}` : value.toFixed(0);

  return (
    <span
      className="mono tnum"
      aria-label={`Elo delta ${signed}`}
      style={{
        display: "inline-flex",
        justifyContent: "center",
        minWidth: 30,
        color,
        padding: "2px 5px",
        borderRadius: 4,
        background,
      }}
    >
      {signed}
    </span>
  );
}

function SimulationMetadata({
  season,
  asOfWeek,
  computedAt,
  nSimulations,
  nSimulationsStatus,
}: {
  season: string | null | undefined;
  asOfWeek: number | null | undefined;
  computedAt: string | null | undefined;
  nSimulations: number | null | undefined;
  nSimulationsStatus?: FieldStatus;
}) {
  return (
    <div
      className="mono"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 3,
        color: "var(--ink-3)",
        fontSize: 10,
      }}
    >
      <span>
        {season ?? "Season unavailable"}
        {asOfWeek != null ? ` · As of Week ${asOfWeek}` : ""}
      </span>

      <span>
        {nSimulations != null ? (
          `${nSimulations.toLocaleString()} simulations`
        ) : nSimulationsStatus === "pending" ? (
          <PendingField placeholder="Simulation count" />
        ) : nSimulationsStatus ? (
          <BlockedField
            blocker={nSimulationsStatus.blocker}
            roadmap={nSimulationsStatus.roadmap}
            placeholder="Simulation count"
          />
        ) : (
          "Simulation count unavailable"
        )}
      </span>

      <span>{formatComputedAt(computedAt)}</span>
    </div>
  );
}

function ProjectionLegend({
  asOfWeek,
  eloStatus,
  clinchedStatus,
}: {
  asOfWeek: number | null | undefined;
  eloStatus?: FieldStatus;
  clinchedStatus?: FieldStatus;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 16,
        marginTop: 14,
        paddingTop: 12,
        borderTop: "1px solid var(--line-soft)",
        color: "var(--ink-4)",
        fontSize: 10,
        flexWrap: "wrap",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span>Lower</span>
        {[0.08, 0.22, 0.38].map((opacity) => (
          <span
            key={opacity}
            aria-hidden="true"
            style={{
              width: 18,
              height: 8,
              borderRadius: 2,
              background: `color-mix(in oklab, var(--pos) ${
                opacity * 100
              }%, transparent)`,
            }}
          />
        ))}
        <span>Higher probability</span>
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {asOfWeek === 1 ? (
          <span>1-week Elo change begins after Week 1.</span>
        ) : eloStatus ? (
          <span>Elo movement unavailable without a prior snapshot.</span>
        ) : null}
        {clinchedStatus === "pending" && (
          <span>Clinched / eliminated status pending</span>
        )}
      </div>
    </div>
  );
}

function matchesTeamFilters(
  metadata: TeamMetadataItem | undefined,
  conference: ConferenceFilter,
  division: DivisionFilter,
): boolean {
  if (conference === "ALL") {
    return division === "ALL";
  }

  if (metadata?.conference !== conference) {
    return false;
  }

  if (
    division !== "ALL" &&
    metadata.division !== division
  ) {
    return false;
  }

  return true;
}

function compareRows(
  a: EnrichedProjection,
  b: EnrichedProjection,
  sortKey: SortKey,
  direction: SortDirection,
): number {
  if (sortKey === "team") {
    const comparison = a.projection.name.localeCompare(
      b.projection.name,
    );
    return direction === "asc" ? comparison : -comparison;
  }

  const aValue = getSortableValue(a, sortKey);
  const bValue = getSortableValue(b, sortKey);

  if (aValue == null && bValue == null) return 0;
  if (aValue == null) return 1;
  if (bValue == null) return -1;

  const comparison = aValue - bValue;
  return direction === "asc" ? comparison : -comparison;
}

function getSortableValue(
  row: EnrichedProjection,
  sortKey: Exclude<SortKey, "team">,
): number | null {
  if (sortKey === "elo") {
    return row.metadata?.rating ?? null;
  }

  return row.projection[sortKey] ?? null;
}

function formatRecord(
  record:
    | {
        wins?: number | null;
        losses?: number | null;
        ties?: number | null;
      }
    | null
    | undefined,
): string {
  if (
    record?.wins == null ||
    record.losses == null
  ) {
    return "N/A";
  }

  if ((record.ties ?? 0) > 0) {
    return `${record.wins}-${record.losses}-${record.ties}`;
  }

  return `${record.wins}-${record.losses}`;
}

function formatDivision(
  conference: string | null | undefined,
  division: string | null | undefined,
): string | null {
  if (!conference || !division) return null;

  const divisionName: Record<string, string> = {
    N: "North",
    S: "South",
    E: "East",
    W: "West",
  };

  return `${conference} ${divisionName[division] ?? division}`;
}

function formatComputedAt(
  computedAt: string | null | undefined,
): string {
  if (!computedAt) return "Computed time unavailable";

  const date = new Date(computedAt);
  if (Number.isNaN(date.getTime())) {
    return "Computed time unavailable";
  }

  return `Computed ${date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  })}`;
}

function FilterSelect({
  id,
  label,
  value,
  options,
  disabled = false,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: Array<{
    value: string;
    label: string;
  }>;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label
      htmlFor={id}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        fontSize: 10,
        color: disabled
          ? "var(--ink-4)"
          : "var(--ink-3)",
      }}
    >
      <span className="upper">{label}</span>

      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        style={{
          minWidth: 138,
          padding: "6px 28px 6px 8px",
          border: "1px solid var(--line-soft)",
          borderRadius: 4,
          background: "var(--bg-2)",
          color: disabled
            ? "var(--ink-4)"
            : "var(--ink-2)",
          fontFamily: "var(--f-sans)",
          fontSize: 11,
          cursor: disabled ? "not-allowed" : "pointer",
        }}
      >
        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
          >
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
