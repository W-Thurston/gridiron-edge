type FilterBarProps = {
  statType: string;
  position: string;
  onStatTypeChange: (value: string) => void;
  onPositionChange: (value: string) => void;
};

const STAT_TYPES = [
  { value: "", label: "All" },
  { value: "qb_pass_yards", label: "QB Pass Yards" },
  { value: "qb_rush_yards", label: "QB Rush Yards" },
  { value: "rb_rush_yards", label: "RB Rush Yards" },
  { value: "wr_rec_yards", label: "WR Rec Yards" },
  { value: "te_rec_yards", label: "TE Rec Yards" },
] as const;

const POSITIONS = [
  { value: "", label: "All" },
  { value: "QB", label: "QB" },
  { value: "RB", label: "RB" },
  { value: "WR", label: "WR" },
  { value: "TE", label: "TE" },
] as const;

export function FilterBar({
  statType,
  position,
  onStatTypeChange,
  onPositionChange,
}: FilterBarProps) {
  return (
    <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
      <FilterSelect
        label="Stat Type"
        value={statType}
        onChange={onStatTypeChange}
        options={STAT_TYPES}
      />
      <FilterSelect
        label="Position"
        value={position}
        onChange={onPositionChange}
        options={POSITIONS}
      />
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: readonly { value: string; label: string }[];
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span className="upper dim2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          background: "var(--bg-1)",
          color: "var(--ink)",
          border: "1px solid var(--line-soft)",
          borderRadius: 5,
          padding: "4px 8px",
          fontSize: 12,
          fontFamily: "var(--f-sans)",
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
