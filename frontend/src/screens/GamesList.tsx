import { useAppState } from "../context/AppStateContext";
import { useGamesList } from "../api/hooks";
import { useNav } from "../context/NavContext";
import { TeamMark } from "../components/primitives/TeamMark";
import { ErrorCard } from "../components/error/ErrorCard";
import { WeeklyComponentValue } from "../components/field-status/WeeklyComponentValue";
import {
  isWeeklyComponentUsable,
  weeklyComponentStatusMessage,
} from "../components/field-status/weeklyComponentStatus";

export function GamesList() {
  const { navigate } = useNav();
  const { state } = useAppState();
  void state; // unused for now; will be used for odds format later
  const { data, isLoading, error, refetch } = useGamesList();

  const handleRowClick = (gameId: string) => {
    navigate("/games", { gameId });
  };

  // Sort by date (ISO strings sort lexically), then game_id as a stable
  // tiebreaker. Time-of-day sort is pending kick_time (ROADMAP §9.7 P0);
  // the API currently exposes game_date only.
  const sortedItems = [...(data?.items ?? [])].sort((a, b) => {
    const da = a.game_date ?? "";
    const db = b.game_date ?? "";
    if (da !== db) return da < db ? -1 : 1;
    return a.game_id < b.game_id ? -1 : a.game_id > b.game_id ? 1 : 0;
  });

  return (
    <div>
      <div
        className="hm-card"
        style={{ padding: 24, marginBottom: 16 }}
      >
        <div
          className="upper dim"
          style={{ fontSize: 10, marginBottom: 12 }}
        >
          Games —{" "}
          {data?.season && data?.week
            ? `${data.season}, Week ${data.week}`
            : "Loading…"}
        </div>

        {isLoading && <div className="dim">Loading…</div>}
        {error && (
          <ErrorCard
            error={error}
            onRetry={() => refetch()}
            title="Couldn't load games"
          />
        )}

        {data && sortedItems.length === 0 && (
          <div className="dim mono" style={{ fontSize: 12 }}>
            No games found for this week.
          </div>
        )}

        {data && sortedItems.length > 0 && (
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
                <th style={{ padding: "8px 12px 8px 0" }}>Date ↑</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Matchup</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Home WP</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Spread</th>
                <th style={{ padding: "8px 0" }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {sortedItems.map((game) => (
                <tr
                  key={game.game_id}
                  className="proj-row"
                  onClick={() => handleRowClick(game.game_id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      handleRowClick(game.game_id);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`View details for ${game.away_team} at ${game.home_team}`}
                  style={{
                    borderTop: "1px solid var(--line-soft)",
                    cursor: "pointer",
                  }}
                >
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}>
                    {game.game_date ?? "—"}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                      }}
                    >
                      <TeamMark abbr={game.away_team} />
                      <span className="dim">@</span>
                      <TeamMark abbr={game.home_team} />
                    </span>
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <WeeklyComponentValue
                      label="Win probability"
                      status={game.win.status}
                      usable={isWeeklyComponentUsable("win", game.win.status)}
                      value={game.win.home_win_prob}
                      format={(value) => `${Math.round(value * 100)}%`}
                      statusMessage={weeklyComponentStatusMessage(
                        "win",
                        game.win.status,
                      )}
                    />
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <WeeklyComponentValue
                      label="Spread"
                      status={game.spread.status}
                      usable={isWeeklyComponentUsable(
                        "spread",
                        game.spread.status,
                      )}
                      value={game.spread.model_spread}
                      format={formatSpread}
                      statusMessage={weeklyComponentStatusMessage(
                        "spread",
                        game.spread.status,
                      )}
                    />
                  </td>
                  <td style={{ padding: "10px 0" }}>
                    <WeeklyComponentValue
                      label="Total"
                      status={game.total.status}
                      usable={isWeeklyComponentUsable("total", game.total.status)}
                      value={game.total.model_total}
                      format={(value) => value.toFixed(1)}
                      statusMessage={weeklyComponentStatusMessage(
                        "total",
                        game.total.status,
                      )}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function formatSpread(spread: number): string {
  const sign = spread > 0 ? "+" : "";
  return `${sign}${spread.toFixed(1)}`;
}
