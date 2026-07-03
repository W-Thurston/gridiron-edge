import { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import { ScreenPlaceholder } from "./ScreenPlaceholder";
import type { components } from "../api/schema";

type CurrentWeek = components["schemas"]["CurrentWeek"];

export function Dashboard() {
  const [data, setData] = useState<CurrentWeek | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const { data, error } = await apiClient.GET("/weeks/current");
      if (error) {
        setError(JSON.stringify(error));
        return;
      }
      if (data) setData(data);
    })();
  }, []);

  return (
    <div>
      <ScreenPlaceholder title="Dashboard" subtitle="/today" />
      <div className="hm-card" style={{ padding: 24, maxWidth: 720, marginTop: 16 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
          API Test — /weeks/current
        </div>
        {error && (
          <div className="neg mono" style={{ fontSize: 12 }}>
            Error: {error}
          </div>
        )}
        {data && (
          <div className="mono tnum" style={{ fontSize: 12 }}>
            <div>Season: {data.season ?? "(null)"}</div>
            <div>Week: {data.week ?? "(null)"}</div>
            <div>Source: {data.source ?? "(null)"}</div>
          </div>
        )}
        {!data && !error && <div className="dim">Loading…</div>}
      </div>
    </div>
  );
}
