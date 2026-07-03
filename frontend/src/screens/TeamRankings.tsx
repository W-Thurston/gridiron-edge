import { useTeamRankings } from "../api/hooks";
import { TeamMark } from "../components/games/TeamMark";
import { PendingField } from "../components/field-status/PendingField";
import { BlockedField } from "../components/field-status/BlockedField";
import type { FieldStatus } from "../components/field-status/types";
import { useNav } from "../context/NavContext";

export function TeamRankings() {
  const { navigate } = useNav();
  const { data, isLoading, error } = useTeamRankings();

  const handleRowClick = (abbr: string) => {
    navigate("/teams", { team: abbr });
  };

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Team Rankings —{" "}
        {data?.season && data?.as_of_week
          ? `${data.season}, through Week ${data.as_of_week}`
          : "Loading…"}
      </div>

      {isLoading && <div className="dim">Loading…</div>}
      {error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {error.message}
        </div>
      )}

      {data && (data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No team ratings found.
        </div>
      )}

      {data && (data.items ?? []).length > 0 && (
        <table
          className="mono tnum"
          style={{
            width: "100%",
            fontSize: 12,
            borderCollapse: "collapse",
          }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>#</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Team</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>Rating</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>Record</th>
              <th style={{ padding: "8px 12px 8px 0" }}>
                <ColumnHeader
                  label="Trend"
                  status={data._meta?.field_status?.["items.trend"] as FieldStatus | undefined}
                />
              </th>
              <th style={{ padding: "8px 12px 8px 0" }}>
                <ColumnHeader
                  label="Off"
                  status={data._meta?.field_status?.["items.off_rating"] as FieldStatus | undefined}
                />
              </th>
              <th style={{ padding: "8px 0" }}>
                <ColumnHeader
                  label="Def"
                  status={data._meta?.field_status?.["items.def_rating"] as FieldStatus | undefined}
                />
              </th>
            </tr>
          </thead>
          <tbody>
            {(data.items ?? []).map((team) => (
              <tr
                key={team.abbr}
                className="proj-row"
                onClick={() => handleRowClick(team.abbr)}
                style={{
                  borderTop: "1px solid var(--line-soft)",
                  cursor: "pointer",
                }}
              >
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right", color: "var(--ink-3)" }}>
                  {team.rank ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <TeamMark abbr={team.abbr} />
                    <span style={{ color: "var(--ink-2)" }}>{team.name}</span>
                  </span>
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {team.rating?.toFixed(0) ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right", color: "var(--ink-2)" }}>
                  {team.record
                    ? `${team.record.wins}-${team.record.losses}${
                        team.record.ties > 0 ? `-${team.record.ties}` : ""
                      }`
                    : "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>—</td>
                <td style={{ padding: "10px 12px 10px 0" }}>—</td>
                <td style={{ padding: "10px 0" }}>—</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ColumnHeader({
  label,
  status,
}: {
  label: string;
  status: FieldStatus | undefined;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      {label}
      {status === "pending" && <PendingField placeholder="" />}
      {status && status !== "pending" && (
        <BlockedField
          blocker={status.blocker}
          roadmap={status.roadmap}
          placeholder=""
        />
      )}
    </span>
  );
}
