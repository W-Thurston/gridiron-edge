type TeamPickerProps = {
  teamA: string;
  teamB: string;
  onTeamAChange: (value: string) => void;
  onTeamBChange: (value: string) => void;
};

const TEAMS = [
  { value: "", label: "Select team…" },
  { value: "ARI", label: "Arizona Cardinals" },
  { value: "ATL", label: "Atlanta Falcons" },
  { value: "BAL", label: "Baltimore Ravens" },
  { value: "BUF", label: "Buffalo Bills" },
  { value: "CAR", label: "Carolina Panthers" },
  { value: "CHI", label: "Chicago Bears" },
  { value: "CIN", label: "Cincinnati Bengals" },
  { value: "CLE", label: "Cleveland Browns" },
  { value: "DAL", label: "Dallas Cowboys" },
  { value: "DEN", label: "Denver Broncos" },
  { value: "DET", label: "Detroit Lions" },
  { value: "GB", label: "Green Bay Packers" },
  { value: "HOU", label: "Houston Texans" },
  { value: "IND", label: "Indianapolis Colts" },
  { value: "JAC", label: "Jacksonville Jaguars" },
  { value: "KAN", label: "Kansas City Chiefs" },
  { value: "LAC", label: "Los Angeles Chargers" },
  { value: "LAR", label: "Los Angeles Rams" },
  { value: "LV", label: "Las Vegas Raiders" },
  { value: "MIA", label: "Miami Dolphins" },
  { value: "MIN", label: "Minnesota Vikings" },
  { value: "NE", label: "New England Patriots" },
  { value: "NO", label: "New Orleans Saints" },
  { value: "NYG", label: "New York Giants" },
  { value: "NYJ", label: "New York Jets" },
  { value: "PHI", label: "Philadelphia Eagles" },
  { value: "PIT", label: "Pittsburgh Steelers" },
  { value: "SEA", label: "Seattle Seahawks" },
  { value: "SF", label: "San Francisco 49ers" },
  { value: "TB", label: "Tampa Bay Buccaneers" },
  { value: "TEN", label: "Tennessee Titans" },
  { value: "WAS", label: "Washington Commanders" },
] as const;

export function TeamPicker({
  teamA,
  teamB,
  onTeamAChange,
  onTeamBChange,
}: TeamPickerProps) {
  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-end" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="upper dim2" style={{ fontSize: 9 }}>
          Team A
        </span>
        <select
          value={teamA}
          onChange={(e) => onTeamAChange(e.target.value)}
          style={selectStyle}
        >
          {TEAMS.map((team) => (
            <option key={team.value} value={team.value}>
              {team.label}
            </option>
          ))}
        </select>
      </div>
      <div
        className="mono dim"
        style={{
          fontSize: 14,
          alignSelf: "flex-end",
          marginBottom: 6,
        }}
      >
        vs
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="upper dim2" style={{ fontSize: 9 }}>
          Team B
        </span>
        <select
          value={teamB}
          onChange={(e) => onTeamBChange(e.target.value)}
          style={selectStyle}
        >
          {TEAMS.map((team) => (
            <option key={team.value} value={team.value}>
              {team.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: "var(--bg-1)",
  color: "var(--ink)",
  border: "1px solid var(--line-soft)",
  borderRadius: 5,
  padding: "4px 8px",
  fontSize: 12,
  fontFamily: "var(--f-sans)",
  minWidth: 180,
};
