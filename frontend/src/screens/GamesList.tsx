import { useAppState } from "../context/AppStateContext";
import { useGamesList } from "../api/hooks";
import { useNav } from "../context/NavContext";
import { ConfidenceTierPill } from "../components/games/ConfidenceTierPill";
import { TeamMark } from "../components/games/TeamMark";
import { WinProbBand } from "../components/games/WinProbBand";

export function GamesList() {
  const { navigate } = useNav();
  const { state } = useAppState();
  void state; // unused for now; will be used for odds format later
  const { data, isLoading, error } = useGamesList();

  const handleRowClick = (gameId: string) => {
    navigate("/games", { gameId });
  };

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
          <div className="neg mono" style={{ fontSize: 12 }}>
            Error: {error.message}
          </div>
        )}

        {data && (data.items ?? []).length === 0 && (
          <div className="dim mono" style={{ fontSize: 12 }}>
            No games found for this week.
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
                <th style={{ padding: "8px 12px 8px 0" }}>Date</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Matchup</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Home WP</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Band</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Spread</th>
                <th style={{ padding: "8px 12px 8px 0" }}>Total</th>
                <th style={{ padding: "8px 0" }}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {(data.items ?? []).map((game) => (
                <tr
                  key={game.game_id}
                  className="proj-row"
                  onClick={() => handleRowClick(game.game_id)}
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
                    {game.prediction?.home_win_prob != null
                      ? `${Math.round(game.prediction.home_win_prob * 100)}%`
                      : "—"}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    <WinProbBand
                      homeWinProb={game.prediction?.home_win_prob}
                      homeWinLo={game.prediction?.home_win_lo}
                      homeWinHi={game.prediction?.home_win_hi}
                    />
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    {game.prediction?.model_spread != null
                      ? formatSpread(game.prediction.model_spread)
                      : "—"}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>
                    {game.prediction?.model_total != null
                      ? game.prediction.model_total.toFixed(1)
                      : "—"}
                  </td>
                  <td style={{ padding: "10px 0" }}>
                    <ConfidenceTierPill
                      tier={game.prediction?.confidence_tier}
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
