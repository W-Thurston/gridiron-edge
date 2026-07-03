import { useCurrentWeek } from "../api/hooks";
import { FieldValue } from "../components/field-status/FieldValue";
import { ScreenPlaceholder } from "./ScreenPlaceholder";

export function Dashboard() {
  const { data, isLoading, error } = useCurrentWeek();

  return (
    <div>
      <ScreenPlaceholder title="Dashboard" subtitle="/today" />

      {/* API loop verification */}
      <div className="hm-card" style={{ padding: 24, maxWidth: 720, marginTop: 16 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
          API — /weeks/current
        </div>
        {isLoading && <div className="dim">Loading…</div>}
        {error && (
          <div className="neg mono" style={{ fontSize: 12 }}>
            Error: {error.message}
          </div>
        )}
        {data && (
          <div className="mono tnum" style={{ fontSize: 12 }}>
            <div>Season: {data.season ?? "(null)"}</div>
            <div>Week: {data.week ?? "(null)"}</div>
            <div>Source: {data.source ?? "(null)"}</div>
          </div>
        )}
      </div>

      {/* Field-status demo */}
      <div className="hm-card" style={{ padding: 24, maxWidth: 720, marginTop: 16 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Field Status Demo — Substep 2.0
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
            <tr style={{ color: "var(--ink-3)" }}>
              <th style={{ textAlign: "left", padding: "4px 12px 4px 0" }}>
                Scenario
              </th>
              <th style={{ textAlign: "left", padding: "4px 12px 4px 0" }}>
                Renders
              </th>
              <th style={{ textAlign: "left", padding: "4px 0" }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            <DemoRow
              scenario="Populated"
              value={<FieldValue value={0.71} />}
              note="Value renders normally"
            />
            <DemoRow
              scenario="Null, no status"
              value={<FieldValue value={null} />}
              note="Bare em dash"
            />
            <DemoRow
              scenario="Null, pending"
              value={<FieldValue value={null} status="pending" />}
              note='Hover "i" badge for tooltip'
            />
            <DemoRow
              scenario="Null, blocked (feature attr.)"
              value={
                <FieldValue
                  value={null}
                  status={{
                    status: "blocked",
                    blocker: "feature_attribution",
                    roadmap: "deferred",
                  }}
                />
              }
              note='Hover "!" badge for tooltip'
            />
            <DemoRow
              scenario="Null, blocked (injuries)"
              value={
                <FieldValue
                  value={null}
                  status={{
                    status: "blocked",
                    blocker: "injury_data_source",
                    roadmap: "§5.3",
                  }}
                />
              }
              note="Different blocker, same shape"
            />
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DemoRow({
  scenario,
  value,
  note,
}: {
  scenario: string;
  value: React.ReactNode;
  note: string;
}) {
  return (
    <tr style={{ borderTop: "1px solid var(--line-soft)" }}>
      <td style={{ padding: "8px 12px 8px 0", color: "var(--ink-2)" }}>
        {scenario}
      </td>
      <td style={{ padding: "8px 12px 8px 0" }}>{value}</td>
      <td style={{ padding: "8px 0", color: "var(--ink-3)", fontSize: 11 }}>
        {note}
      </td>
    </tr>
  );
}
