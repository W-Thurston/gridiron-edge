import { useMemo, useState } from "react";
import { useProjections } from "../api/hooks";
import { useTeamMetadata } from "../api/team_metadata_hook";
import { ErrorCard } from "../components/error/ErrorCard";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { StatusPill } from "../components/projections/StatusPill";
import { HeatCell } from "../components/primitives/HeatCell";
import { Pill } from "../components/primitives/Pill";
import {
  SortableHeader,
  type SortDirection,
} from "../components/primitives/SortableHeader";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";

type ConferenceFilter = "ALL" | "AFC" | "NFC";

type SortKey =
  | "team"
  | "avg_wins"
  | "make_playoffs"
  | "reach_div"
  | "reach_conf"
  | "reach_sb"
  | "win_sb";

type ProjectionItem = NonNullable<
  NonNullable<ReturnType<typeof useProjections>["data"]>["items"]
>[number];

type TeamMetadataItem = NonNullable<
  NonNullable<ReturnType<typeof useTeamMetadata>["data"]>["items"]
>[number];

type EnrichedProjection = {
  projection: ProjectionItem;
  metadata: TeamMetadataItem | undefined;
};

export function PlayoffProjections() {
  const { data, isLoading, error, refetch } = useProjections();
  const { data: teamMetadata } = useTeamMetadata();
  const { navigate } = useNav();

  const [conference, setConference] =
    useState<ConferenceFilter>("ALL");
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

    const filtered = rows.filter(({ metadata }) => {
      if (conference === "ALL") return true;
      return metadata?.conference === conference;
    });

    return filtered.toSorted((a, b) =>
      compareRows(a, b, sortKey, sortDirection),
    );
  }, [
    conference,
    data?.items,
    metadataByAbbr,
    sortDirection,
    sortKey,
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

  const fieldStatus = data?._meta?.field_status;

  return (
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
          computedAt={data?.computed_at}
          nSimulations={data?.n_simulations}
          nSimulationsStatus={
            fieldStatus?.n_simulations as FieldStatus | undefined
          }
        />
      </header>

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
          aria-label="Conference filter"
          style={{ display: "flex", gap: 6 }}
        >
          {(["ALL", "AFC", "NFC"] as const).map((value) => (
            <Pill
              key={value}
              active={conference === value}
              onClick={() => setConference(value)}
            >
              {value === "ALL" ? "All" : value}
            </Pill>
          ))}
        </div>

        {data && (
          <div className="mono dim" style={{ fontSize: 10 }}>
            {visibleRows.length} of {data.total ?? data.items?.length ?? 0} teams
          </div>
        )}
      </div>

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
                minWidth: 810,
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
                      <TeamIdentity
                        projection={projection}
                        metadata={metadata}
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
                        eliminatedStatus={
                          fieldStatus?.["items.eliminated"] as
                            | FieldStatus
                            | undefined
                        }
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
                      {projection.avg_wins?.toFixed(1) ?? "N/A"}
                    </td>

                    <ProbabilityCell
                      value={projection.make_playoffs}
                      label={`${projection.name} make playoffs`}
                      status={
                        fieldStatus?.["items.make_playoffs"] as
                          | FieldStatus
                          | undefined
                      }
                    />

                    <ProbabilityCell
                      value={projection.reach_div}
                      label={`${projection.name} reach divisional round`}
                      status={
                        fieldStatus?.["items.reach_div"] as
                          | FieldStatus
                          | undefined
                      }
                    />

                    <ProbabilityCell
                      value={projection.reach_conf}
                      label={`${projection.name} reach conference championship`}
                      status={
                        fieldStatus?.["items.reach_conf"] as
                          | FieldStatus
                          | undefined
                      }
                    />

                    <ProbabilityCell
                      value={projection.reach_sb}
                      label={`${projection.name} reach Super Bowl`}
                      status={
                        fieldStatus?.["items.reach_sb"] as
                          | FieldStatus
                          | undefined
                      }
                    />

                    <ProbabilityCell
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
    </div>
  );
}

function ProbabilityCell({
  value,
  label,
  status,
}: {
  value: number | null | undefined;
  label: string;
  status?: FieldStatus;
}) {
  return (
    <td
      style={{
        padding: "10px 12px 10px 0",
        textAlign: "right",
      }}
    >
      <HeatCell value={value} label={label} status={status} />
    </td>
  );
}

function TeamIdentity({
  projection,
  metadata,
  eloStatus,
  onNavigate,
}: {
  projection: ProjectionItem;
  metadata: TeamMetadataItem | undefined;
  eloStatus?: FieldStatus;
  clinchedStatus?: FieldStatus;
  eliminatedStatus?: FieldStatus;
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
      <TeamMark abbr={projection.abbr} />

      <div style={{ minWidth: 0, flex: 1 }}>
        <button
          type="button"
          onClick={onNavigate}
          style={{
            display: "block",
            width: "100%",
            padding: 0,
            border: 0,
            background: "transparent",
            color: "var(--ink-2)",
            font: "inherit",
            textAlign: "left",
            cursor: "pointer",
          }}
        >
          {projection.name}
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
          <span>{division ?? "Metadata unavailable"}</span>

          <EloDelta
            value={projection.elo_delta}
            status={eloStatus}
          />

          <StatusPill
            clinched={projection.clinched}
            eliminated={projection.eliminated}
          />
        </div>
      </div>
    </div>
  );
}

function EloDelta({
  value,
  status,
}: {
  value: number | null | undefined;
  status?: FieldStatus;
}) {
  if (value == null) {
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
      <span title="Elo delta not available">
        Elo Δ N/A
      </span>
    );
  }

  const color =
    value > 0
      ? "var(--pos)"
      : value < 0
        ? "var(--neg)"
        : "var(--ink-3)";

  const signed = value > 0 ? `+${value.toFixed(0)}` : value.toFixed(0);

  return (
    <span
      className="mono tnum"
      aria-label={`Elo delta ${signed}`}
      style={{
        color,
        padding: "1px 4px",
        borderRadius: 3,
        background:
          value > 0
            ? "color-mix(in oklab, var(--pos) 10%, transparent)"
            : value < 0
              ? "color-mix(in oklab, var(--neg) 10%, transparent)"
              : "var(--bg-2)",
      }}
    >
      Elo {signed}
    </span>
  );
}

function SimulationMetadata({
  season,
  computedAt,
  nSimulations,
  nSimulationsStatus,
}: {
  season: string | null | undefined;
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
      <span>{season ?? "Season unavailable"}</span>

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
  eloStatus,
  clinchedStatus,
}: {
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
        {eloStatus && (
          <span>Elo movement unavailable without a prior snapshot</span>
        )}
        {clinchedStatus === "pending" && (
          <span>Clinched / eliminated status pending</span>
        )}
      </div>
    </div>
  );
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

  const aValue = a.projection[sortKey];
  const bValue = b.projection[sortKey];

  if (aValue == null && bValue == null) return 0;
  if (aValue == null) return 1;
  if (bValue == null) return -1;

  const comparison = Number(aValue) - Number(bValue);
  return direction === "asc" ? comparison : -comparison;
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
