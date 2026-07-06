type SparkProps = {
  /** Data points as numbers. Rendered left-to-right, scaled to min/max. */
  data: number[];
  /** SVG width in px. Default 80. */
  width?: number;
  /** SVG height in px. Default 22. */
  height?: number;
  /** Line color. Accepts CSS variables and hex strings. Default var(--accent). */
  color?: string;
  /** Line stroke width in px. Default 1.5. */
  strokeWidth?: number;
};

/**
 * Generic sparkline. Renders a small SVG polyline for a time-series
 * `number[]` scaled to the container. Used for compact trend
 * visualizations across screens (rating trends, model performance,
 * player game logs, etc.).
 *
 * Handles edge cases:
 * - Empty array → returns null
 * - Single point → renders a small dot at center
 * - Constant values → renders a flat horizontal line at vertical center
 */
export function Spark({
  data,
  width = 80,
  height = 22,
  color = "var(--accent)",
  strokeWidth = 1.5,
}: SparkProps) {
  if (!data || data.length === 0) return null;

  // Single point: render a small dot at the center
  if (data.length === 1) {
    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        <circle cx={width / 2} cy={height / 2} r={2} fill={color} />
      </svg>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1; // guard against div-by-zero for flat data

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return [x, y];
  });

  const pathD = pts
    .map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1))
    .join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={pathD} fill="none" stroke={color} strokeWidth={strokeWidth} />
    </svg>
  );
}
