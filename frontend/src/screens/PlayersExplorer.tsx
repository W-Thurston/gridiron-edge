import { useState } from "react";
import { usePropsList } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/games/TeamMark";
import { FilterBar } from "../components/props/FilterBar";
import { useNav } from "../context/NavContext";

export function PlayersExplorer() {
  const { navigate } = useNav();
  const [statType, setStatType] = useState("");
  const [position, setPosition] = useState("");

  const { data, isLoading, error } = usePropsList({
    stat_type: statType || undefined,
    position: position || undefined,
  });

  const handleRowClick = (propId: string) => {
    navigate("/players", { propId });
  };

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Players — Prop Projections
        {data?.season && data?.week
          ? ` (${data.season}, Week ${data.week})`
          : ""}
      </div>

      <FilterBar
        statType={statType}
        position={position}
        onStatTypeChange={setStatType}
        onPositionChange={setPosition}
      />

      {isLoading && <div className="dim">Loading…</div>}
      {error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {error.message}
        </div>
      )}

      {data && (data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No prop projections found.
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
              <th style={{ padding: "8px 12px 8px 0" }}>Player</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Team</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Stat</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Proj. Mean
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Std
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                Band (Lo–Hi)
              </th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>
                <ColumnHeader
                  label="Line"
                  status={
                    data._meta?.field_status?.["items.line_context.line"] as
                      | FieldStatus
                      | undefined
                  }
                />
              </th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>
                <ColumnHeader
                  label="P(Over)"
                  status={
                    data._meta?.field_status?.["items.line_context.p_over"] as
                      | FieldStatus
                      | undefined
                  }
                />
              </th>
            </tr>
          </thead>
          <tbody>
            {(data.items ?? []).map((prop) => (
              <tr
                key={prop.prop_id}
                className="proj-row"
                onClick={() => handleRowClick(prop.prop_id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    handleRowClick(prop.prop_id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-label={`View prop details for ${prop.player_name} ${prop.stat_type}`}
                style={{
                  borderTop: "1px solid var(--line-soft)",
                  cursor: "pointer",
                }}
              >
                <td style={{ padding: "10px 12px 10px 0" }}>
                  {prop.player_name}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>
                  <TeamMark abbr={prop.team} />
                </td>
                <td
                  style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}
                >
                  {prop.stat_type}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {prop.projection?.predicted_mean?.toFixed(1) ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {prop.projection?.predicted_std?.toFixed(1) ?? "—"}
                </td>
                <td
                  style={{
                    padding: "10px 12px 10px 0",
                    textAlign: "right",
                    color: "var(--ink-3)",
                  }}
                >
                  {prop.projection?.lo_90 != null &&
                  prop.projection?.hi_90 != null
                    ? `${prop.projection.lo_90.toFixed(0)}–${prop.projection.hi_90.toFixed(0)}`
                    : "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  —
                </td>
                <td style={{ padding: "10px 0", textAlign: "right" }}>—</td>
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
        justifyContent: "flex-end",
      }}
    >
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
