import { useProjections } from "../api/hooks";
import { TeamMark } from "../components/games/TeamMark";
import { PendingField } from "../components/field-status/PendingField";
import { BlockedField } from "../components/field-status/BlockedField";
import type { FieldStatus } from "../components/field-status/types";
import { StatusPill } from "../components/projections/StatusPill";

export function PlayoffProjections() {
  const { data, isLoading, error } = useProjections();

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Playoff Projections
        </div>
        <div
          className="mono dim2"
          style={{ fontSize: 11, display: "flex", gap: 12, alignItems: "center" }}
        >
          <NSimulationsField
            data={data}
            status={data?._meta?.field_status?.n_simulations as FieldStatus | undefined}
          />
        </div>
      </div>

      {isLoading && <div className="dim">Loading…</div>}
      {error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {error.message}
        </div>
      )}

      {data && (data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No projections found. Run `gridiron sim run` to populate.
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
              <th style={{ padding: "8px 12px 8px 0" }}>Team</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Avg Wins
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Playoffs
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Div.
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Conf.
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Reach SB
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Win SB
              </th>
              <th style={{ padding: "8px 12px 8px 0" }}>
                <ColumnHeader
                  label="1w Δ"
                  status={
                    data._meta?.field_status?.["items.week_over_week_delta"] as
                      | FieldStatus
                      | undefined
                  }
                />
              </th>
              <th style={{ padding: "8px 0" }}>
                <ColumnHeader
                  label="Status"
                  status={
                    data._meta?.field_status?.["items.clinched"] as
                      | FieldStatus
                      | undefined
                  }
                />
              </th>
            </tr>
          </thead>
          <tbody>
            {(data.items ?? []).map((team) => (
              <tr
                key={team.abbr}
                className="proj-row"
                style={{
                  borderTop: "1px solid var(--line-soft)",
                }}
              >
                <td style={{ padding: "10px 12px 10px 0" }}>
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    <TeamMark abbr={team.abbr} />
                    <span style={{ color: "var(--ink-2)" }}>{team.name}</span>
                  </span>
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {team.avg_wins?.toFixed(1) ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {formatPct(team.make_playoffs)}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {formatPct(team.reach_div)}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {formatPct(team.reach_conf)}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {formatPct(team.reach_sb)}
                </td>
                <td
                  style={{
                    padding: "10px 12px 10px 0",
                    textAlign: "right",
                    color: "var(--pos)",
                  }}
                >
                  {formatPct(team.win_sb)}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>—</td>
                <td style={{ padding: "10px 0" }}>
                  <StatusPill
                    clinched={team.clinched}
                    eliminated={team.eliminated}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function formatPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${Math.round(v * 100)}%`;
}

function NSimulationsField({
  data,
  status,
}: {
  data: { n_simulations?: number | null } | undefined;
  status: FieldStatus | undefined;
}) {
  const n = data?.n_simulations;
  if (n != null) {
    return <span>{`${n.toLocaleString()} sims`}</span>;
  }
  if (status === "pending") {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
        <span className="dim2">n_simulations</span>
        <PendingField placeholder="" />
      </span>
    );
  }
  if (status) {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
        <span className="dim2">n_simulations</span>
        <BlockedField
          blocker={status.blocker}
          roadmap={status.roadmap}
          placeholder=""
        />
      </span>
    );
  }
  return null;
}

function ColumnHeader({
  label,
  status,
}: {
  label: string;
  status: FieldStatus | undefined;
}) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      {label}
      <ColumnHeaderStatus status={status} />
    </span>
  );
}

function ColumnHeaderStatus({ status }: { status: FieldStatus | undefined }) {
  if (!status) return null;
  if (status === "pending") return <PendingField placeholder="" />;
  return (
    <BlockedField
      blocker={status.blocker}
      roadmap={status.roadmap}
      placeholder=""
    />
  );
}
