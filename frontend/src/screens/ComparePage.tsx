import { useEffect, useState } from "react";
import { useCompareTeams } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/games/TeamMark";
import { TeamPicker } from "../components/compare/TeamPicker";
import { useNav } from "../context/NavContext";

export function ComparePage() {
  const { route, navigate } = useNav();
  const initialTeamA = route.params.team_a ?? "";
  const initialTeamB = route.params.team_b ?? "";

  const [teamA, setTeamA] = useState(initialTeamA);
  const [teamB, setTeamB] = useState(initialTeamB);

  // Sync selections to URL for bookmarking.
  useEffect(() => {
    const params: Record<string, string> = {};
    if (teamA) params.team_a = teamA;
    if (teamB) params.team_b = teamB;
    navigate("/compare", params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teamA, teamB]);

  const { data, isLoading, error } = useCompareTeams({
    team_a: teamA || null,
    team_b: teamB || null,
  });

  const bothSelected = teamA && teamB;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Team Comparison
      </div>

      <div style={{ marginBottom: 16 }}>
        <TeamPicker
          teamA={teamA}
          teamB={teamB}
          onTeamAChange={setTeamA}
          onTeamBChange={setTeamB}
        />
      </div>

      {!bothSelected && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          Select two teams to compare.
        </div>
      )}

      {bothSelected && isLoading && <div className="dim">Loading…</div>}
      {bothSelected && error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {error.message}
        </div>
      )}

      {bothSelected && data && (
        <div>
          <div
            className="mono dim"
            style={{ fontSize: 11, marginBottom: 12 }}
          >
            Season: {data.season ?? "—"}
          </div>

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
                <th style={{ padding: "8px 12px 8px 0" }}>Stat</th>
                <th
                  style={{
                    padding: "8px 12px 8px 0",
                    textAlign: "right",
                    fontWeight: 500,
                    color: "var(--ink)",
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 8,
                      justifyContent: "flex-end",
                    }}
                  >
                    <TeamMark abbr={data.team_a} />
                    {data.team_a}
                  </span>
                </th>
                <th
                  style={{
                    padding: "8px 0",
                    textAlign: "right",
                    fontWeight: 500,
                    color: "var(--ink)",
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 8,
                      justifyContent: "flex-end",
                    }}
                  >
                    <TeamMark abbr={data.team_b} />
                    {data.team_b}
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {(data.stats ?? []).map((row) => (
                <StatRowDisplay
                  key={row.key}
                  row={row}
                  status={
                    data._meta?.field_status?.[row.key] as
                      | FieldStatus
                      | undefined
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

type StatRow = {
  key: string;
  label: string;
  unit?: string | null;
  team_a_value?: number | string | null;
  team_b_value?: number | string | null;
};

function StatRowDisplay({
  row,
  status,
}: {
  row: StatRow;
  status: FieldStatus | undefined;
}) {
  return (
    <tr style={{ borderTop: "1px solid var(--line-soft)" }}>
      <td
        style={{
          padding: "10px 12px 10px 0",
          color: "var(--ink-2)",
        }}
      >
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          {row.label}
          <StatRowLabelStatus status={status} />
        </span>
      </td>
      <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
        <CompareCell value={row.team_a_value} status={status} />
      </td>
      <td style={{ padding: "10px 0", textAlign: "right" }}>
        <CompareCell value={row.team_b_value} status={status} />
      </td>
    </tr>
  );
}

function StatRowLabelStatus({ status }: { status: FieldStatus | undefined }) {
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

function CompareCell({
  value,
  status,
}: {
  value: number | string | null | undefined;
  status: FieldStatus | undefined;
}) {
  if (value != null && value !== "") {
    return <>{typeof value === "number" ? formatNumericValue(value) : value}</>;
  }
  // Row is field_status-scaffolded; just show em dash (label already shows badge).
  if (status) return <span className="dim2">—</span>;
  return <span className="dim2">—</span>;
}

function formatNumericValue(v: number): string {
  // Elo-style ratings render as whole numbers, other numerics get one decimal.
  if (Math.abs(v) > 100) return v.toFixed(0);
  return v.toFixed(1);
}
