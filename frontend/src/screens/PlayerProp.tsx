import { useComparePlayer, useProp } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { TeamMark } from "../components/games/TeamMark";
import { useNav } from "../context/NavContext";

export function PlayerProp() {
  const { route, navigate } = useNav();
  const propId = route.params.propId ?? null;

  const propResult = useProp(propId);
  const compareResult = useComparePlayer(propId);

  if (!propId) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">No prop selected.</div>
      </div>
    );
  }

  // Render back nav + status-dependent content.
  const backNav = (
    <div>
      <button
        type="button"
        onClick={() => navigate("/players")}
        className="dim mono"
        style={{
          background: "transparent",
          border: "none",
          padding: 0,
          cursor: "pointer",
          font: "inherit",
          color: "var(--ink-3)",
          fontSize: 12,
        }}
      >
        ← Players
      </button>
    </div>
  );

  if (propResult.isLoading || compareResult.isLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">Loading…</div>
        </div>
      </div>
    );
  }

  if (propResult.error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="neg mono" style={{ marginBottom: 8 }}>
            Error: {propResult.error.message}
          </div>
          <div className="dim mono" style={{ fontSize: 11 }}>
            This prop may not exist in the archive.
          </div>
        </div>
      </div>
    );
  }

  const prop = propResult.data;
  const compare = compareResult.data;

  if (!prop) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <div className="hm-card" style={{ padding: 24 }}>
          <div className="dim">No prop data available.</div>
        </div>
      </div>
    );
  }

  const propStatus = prop._meta?.field_status;
  const compareStatus = compare?._meta?.field_status;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {backNav}

      {/* Identity card */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Player Prop — {prop.prop_id}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 20,
          }}
        >
          <TeamMark abbr={prop.team} />
          <span>{prop.player_name}</span>
          <span className="dim mono" style={{ fontSize: 14 }}>
            {prop.position} · {prop.stat_type}
          </span>
        </div>
        <div
          className="mono dim"
          style={{ fontSize: 12, marginTop: 12, display: "flex", gap: 16 }}
        >
          <span>Season: {prop.season ?? "—"}</span>
          <span>Week: {prop.week ?? "—"}</span>
          <span>Model: {prop.model_key}</span>
        </div>
      </div>

      {/* Projection card */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Projection
        </div>
        {prop.projection ? (
          <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
            <ProjectionCell
              label="Predicted Mean"
              value={prop.projection.predicted_mean?.toFixed(1) ?? "—"}
            />
            <ProjectionCell
              label="Std (Uncertainty)"
              value={prop.projection.predicted_std?.toFixed(1) ?? "—"}
            />
            <ProjectionCell
              label="10th %ile"
              value={prop.projection.lo_90?.toFixed(0) ?? "—"}
            />
            <ProjectionCell
              label="90th %ile"
              value={prop.projection.hi_90?.toFixed(0) ?? "—"}
            />
          </div>
        ) : (
          <div className="dim mono">No projection available.</div>
        )}
      </div>

      {/* Line context card */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Line Context
        </div>
        <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
          <ProjectionCell
            label="Line"
            value={
              <InlineFieldValue
                value={prop.line_context?.line}
                formatter={(v) => v.toFixed(1)}
                status={propStatus?.["line_context.line"] as FieldStatus | undefined}
              />
            }
          />
          <ProjectionCell
            label="P(Over)"
            value={
              <InlineFieldValue
                value={prop.line_context?.p_over}
                formatter={(v) => `${(v * 100).toFixed(0)}%`}
                status={propStatus?.["line_context.p_over"] as FieldStatus | undefined}
              />
            }
          />
          <ProjectionCell
            label="Lean"
            value={
              <InlineFieldValue
                value={prop.line_context?.lean}
                formatter={(v) => v}
                status={propStatus?.["line_context.lean"] as FieldStatus | undefined}
              />
            }
          />
          <ProjectionCell
            label="Confidence"
            value={
              <InlineFieldValue
                value={prop.line_context?.confidence_tier}
                formatter={(v) => v}
                status={propStatus?.["line_context.confidence_tier"] as FieldStatus | undefined}
              />
            }
          />
        </div>
      </div>

      {/* Player-vs-Defense compare card */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Player vs Defense
        </div>
        {compare && (compare.stats ?? []).length > 0 ? (
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
                  style={{ padding: "8px 12px 8px 0", textAlign: "right" }}
                >
                  Projection
                </th>
                <th style={{ padding: "8px 0", textAlign: "right" }}>Defense</th>
              </tr>
            </thead>
            <tbody>
              {(compare.stats ?? []).map((row) => (
                <tr
                  key={row.key}
                  style={{ borderTop: "1px solid var(--line-soft)" }}
                >
                  <td
                    style={{
                      padding: "10px 12px 10px 0",
                      color: "var(--ink-2)",
                    }}
                  >
                    {row.label}
                  </td>
                  <td
                    style={{
                      padding: "10px 12px 10px 0",
                      textAlign: "right",
                    }}
                  >
                    <CompareCell value={row.projection_value} />
                  </td>
                  <td style={{ padding: "10px 0", textAlign: "right" }}>
                    <CompareCell
                      value={row.defense_value}
                      status={
                        compareStatus?.[row.key] as FieldStatus | undefined
                      }
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="dim mono">No comparison data available.</div>
        )}
      </div>

      {/* Scaffolded cards from PropDetail */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <ComingSoonCard
          title="Historical vs Opponent"
          status={propStatus?.historical_vs_opponent as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Situational Splits"
          status={propStatus?.situational_splits as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Recent Form"
          status={propStatus?.recent_form as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Injury Status"
          status={propStatus?.injury_status as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Prop Reasoning"
          status={propStatus?.prop_reasoning as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Multi-Book Shopping"
          status={propStatus?.multi_book_shopping as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

function ProjectionCell({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div style={{ minWidth: 100 }}>
      <div className="upper dim2" style={{ fontSize: 10, marginBottom: 6 }}>
        {label}
      </div>
      <div className="mono tnum" style={{ fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}

function InlineFieldValue<T>({
  value,
  formatter,
  status,
}: {
  value: T | null | undefined;
  formatter: (v: T) => string;
  status: FieldStatus | undefined;
}) {
  if (value != null && value !== "") {
    return <>{formatter(value)}</>;
  }
  if (!status) return <span className="dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

function CompareCell({
  value,
  status,
}: {
  value: number | string | null | undefined;
  status?: FieldStatus | undefined;
}) {
  if (value != null && value !== "") {
    return <>{typeof value === "number" ? value.toFixed(1) : value}</>;
  }
  if (!status) return <span className="dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

function ComingSoonCard({
  title,
  status,
}: {
  title: string;
  status: FieldStatus | undefined;
}) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        className="upper dim"
        style={{
          fontSize: 10,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>{title}</span>
        <ComingSoonStatus status={status} />
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Not yet available
      </div>
    </div>
  );
}

function ComingSoonStatus({ status }: { status: FieldStatus | undefined }) {
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
