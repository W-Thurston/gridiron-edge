import { useCurrentWeek } from "../api/hooks";
import { ScreenPlaceholder } from "./ScreenPlaceholder";

export function Dashboard() {
  const { data, isLoading, error } = useCurrentWeek();

  return (
    <div>
      <ScreenPlaceholder title="Dashboard" subtitle="/today" />
      <div className="hm-card" style={{ padding: 24, maxWidth: 720, marginTop: 16 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
          API Test — /weeks/current
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
    </div>
  );
}
