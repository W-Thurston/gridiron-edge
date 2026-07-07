type RatingHistoryPoint = {
  week: number;
  rating: number;
};

type RecentResult = {
  week: number;
  result?: string | null;
};

type RatingChartProps = {
  history?: RatingHistoryPoint[] | null;
  recentResults?: RecentResult[] | null;
  /** Chart height in px. Default 160. */
  height?: number;
  /** Color for the line. Default var(--pos). */
  color?: string;
};

/**
 * Rating chart component. Renders SVG line chart with:
 * - Y-axis grid + labels
 * - Line for rating series
 * - Small dots at each data point
 * - X-axis week labels (every ~4-5 weeks)
 * - W/L markers below X-axis when recentResults provided
 *
 * Responsive width via SVG viewBox scaling.
 */
export function RatingChart({
  history,
  recentResults,
  height = 160,
  color = "var(--pos)",
}: RatingChartProps) {
  if (!history || history.length === 0) {
    return (
      <div
        className="dim"
        style={{
          padding: 20,
          textAlign: "center",
          fontSize: 12,
        }}
      >
        No rating history available.
      </div>
    );
  }

  // Chart dimensions (viewBox space)
  const width = 800;
  const pad = { top: 12, right: 24, bottom: 24, left: 40 };

  // Compute Y-axis range with padding
  const ratings = history.map((p) => p.rating);
  const minR = Math.min(...ratings);
  const maxR = Math.max(...ratings);
  const range = Math.max(maxR - minR, 20);
  const yMin = Math.floor((minR - range * 0.1) / 5) * 5;
  const yMax = Math.ceil((maxR + range * 0.1) / 5) * 5;

  // Y-axis grid values (4-5 lines)
  const yStep = (yMax - yMin) / 4;
  const yGridValues = [0, 1, 2, 3, 4].map((i) => yMin + i * yStep);

  // Chart body dimensions
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  // X-axis: label every ~4-5 weeks
  const xLabelStep = Math.max(1, Math.floor(history.length / 5));
  const xLabels = history
    .map((p, i) => (i % xLabelStep === 0 || i === history.length - 1 ? p : null))
    .filter((p): p is RatingHistoryPoint => p !== null);

  // Point positioning
  const x = (i: number) => pad.left + (i / Math.max(history.length - 1, 1)) * chartW;
  const y = (rating: number) =>
    pad.top + chartH - ((rating - yMin) / (yMax - yMin)) * chartH;

  // Build path
  const pathD = history
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)},${y(p.rating).toFixed(1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height + 16}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {/* Y-axis grid */}
      {yGridValues.map((value) => (
        <g key={value}>
          <line
            x1={pad.left}
            y1={y(value)}
            x2={width - pad.right}
            y2={y(value)}
            stroke="var(--line-soft)"
            strokeWidth={0.5}
          />
          <text
            x={pad.left - 6}
            y={y(value)}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={9}
            fontFamily="var(--f-mono)"
            fill="var(--ink-4)"
          >
            {value.toFixed(0)}
          </text>
        </g>
      ))}

      {/* Rating line */}
      <path
        d={pathD}
        stroke={color}
        strokeWidth={1.5}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Data point dots */}
      {history.map((p, i) => (
        <circle key={p.week} cx={x(i)} cy={y(p.rating)} r={2} fill={color} />
      ))}

      {/* X-axis labels */}
      {xLabels.map((p) => {
        const i = history.findIndex((h) => h.week === p.week);
        return (
          <text
            key={p.week}
            x={x(i)}
            y={height - 6}
            textAnchor="middle"
            fontSize={9}
            fontFamily="var(--f-mono)"
            fill="var(--ink-4)"
          >
            Wk {p.week}
          </text>
        );
      })}

      {/* W/L markers below x-axis */}
      {recentResults &&
        recentResults.map((r) => {
          const i = history.findIndex((h) => h.week === r.week);
          if (i === -1 || r.result == null) return null;
          const markerColor =
            r.result === "W"
              ? "var(--pos)"
              : r.result === "L"
                ? "var(--neg)"
                : "var(--ink-3)";
          return (
            <text
              key={`marker-${r.week}`}
              x={x(i)}
              y={height + 4}
              textAnchor="middle"
              fontSize={9}
              fontFamily="var(--f-mono)"
              fill={markerColor}
              fontWeight={600}
            >
              {r.result}
            </text>
          );
        })}
    </svg>
  );
}
