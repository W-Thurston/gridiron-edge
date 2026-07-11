type Bar = {
  label: string; // x-axis label (e.g., "1", "12")
  value: number | null;
};

type BarChartProps = {
  bars: Bar[];
  /** Solid horizontal reference line (e.g., team-allowed average). */
  referenceValue?: number | null;
  /** Label for the reference line's right-edge tag. */
  referenceLabel?: string;
  height?: number;
  barColor?: string;
  referenceColor?: string;
};

/**
 * SVG bar chart with an optional solid horizontal reference line.
 *
 * Used by Compare Player-vs-Defense (per-game stat + team-allowed line)
 * and reusable for PlayerProp's game-log chart.
 *
 * Y-scale spans 0 to max(bar values, reference) × 1.1 so both fit. Bars
 * with null value render as gaps. Responsive width via viewBox.
 */
export function BarChart({
  bars,
  referenceValue,
  referenceLabel,
  height = 220,
  barColor = "var(--pos)",
  referenceColor = "var(--info)",
}: BarChartProps) {
  if (!bars || bars.length === 0) {
    return (
      <div
        className="dim"
        style={{ padding: 20, textAlign: "center", fontSize: 12 }}
      >
        No game history available.
      </div>
    );
  }

  const width = 800;
  const pad = { top: 16, right: 48, bottom: 28, left: 40 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const values = bars.map((b) => b.value ?? 0);
  const maxData = Math.max(...values, referenceValue ?? 0, 1);
  const yMax = maxData * 1.1;

  const toY = (v: number) => pad.top + chartH - (v / yMax) * chartH;

  const slot = chartW / bars.length;
  const barW = Math.max(2, slot * 0.6);

  // Y-grid: 4 lines.
  const yGrid = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {/* Y grid + labels */}
      {yGrid.map((v) => (
        <g key={v}>
          <line
            x1={pad.left}
            y1={toY(v)}
            x2={width - pad.right}
            y2={toY(v)}
            stroke="var(--line-soft)"
            strokeWidth={0.5}
          />
          <text
            x={pad.left - 6}
            y={toY(v)}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={9}
            fontFamily="var(--f-mono)"
            fill="var(--ink-4)"
          >
            {v.toFixed(0)}
          </text>
        </g>
      ))}

      {/* Bars */}
      {bars.map((b, i) => {
        if (b.value == null) return null;
        const x = pad.left + i * slot + (slot - barW) / 2;
        const y = toY(b.value);
        const h = pad.top + chartH - y;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barW}
            height={Math.max(0, h)}
            rx={1.5}
            fill={barColor}
            opacity={0.85}
          />
        );
      })}

      {/* Reference line (team-allowed average) */}
      {referenceValue != null && (
        <>
          <line
            x1={pad.left}
            y1={toY(referenceValue)}
            x2={width - pad.right}
            y2={toY(referenceValue)}
            stroke={referenceColor}
            strokeWidth={1.5}
          />
          <text
            x={width - pad.right + 4}
            y={toY(referenceValue)}
            dominantBaseline="central"
            fontSize={9}
            fontFamily="var(--f-mono)"
            fill={referenceColor}
            fontWeight={600}
          >
            {referenceValue.toFixed(0)}
          </text>
          {referenceLabel && (
            <text
              x={width - pad.right + 4}
              y={toY(referenceValue) + 11}
              dominantBaseline="central"
              fontSize={7.5}
              fontFamily="var(--f-mono)"
              fill="var(--ink-4)"
            >
              {referenceLabel}
            </text>
          )}
        </>
      )}

      {/* X labels */}
      {bars.map((b, i) => {
        const x = pad.left + i * slot + slot / 2;
        return (
          <text
            key={`x-${i}`}
            x={x}
            y={height - 8}
            textAnchor="middle"
            fontSize={8.5}
            fontFamily="var(--f-mono)"
            fill="var(--ink-4)"
          >
            {b.label}
          </text>
        );
      })}
    </svg>
  );
}
