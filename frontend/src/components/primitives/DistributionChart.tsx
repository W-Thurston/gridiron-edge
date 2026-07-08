type DistributionChartProps = {
  /** Central tendency (model prediction). Renders as vertical line. */
  mean?: number | null;
  /** Standard deviation. Drives the width of the distribution. */
  std?: number | null;
  /** 10th percentile — start of credible band. */
  lo?: number | null;
  /** 90th percentile — end of credible band. */
  hi?: number | null;
  /** Optional book line marker (currently unused; line data pending). */
  line?: number | null;
  /** Chart height in px. Default 200. */
  height?: number;
  /** Color for the distribution + mean line. Default var(--pos). */
  color?: string;
};

/**
 * Renders a Gaussian probability density curve from mean + std with
 * optional 90% credible band shading and mean marker.
 *
 * Used by PlayerProp and Compare (Player vs Defense mode).
 *
 * Responsive width via SVG viewBox scaling; consumer sets container width.
 *
 * Fallback: shows placeholder text if mean or std unavailable.
 */
export function DistributionChart({
  mean,
  std,
  lo,
  hi,
  line,
  height = 200,
  color = "var(--pos)",
}: DistributionChartProps) {
  if (mean == null || std == null || std <= 0) {
    return (
      <div
        className="dim"
        style={{
          padding: 20,
          textAlign: "center",
          fontSize: 12,
        }}
      >
        No distribution data available.
      </div>
    );
  }

  // Chart dimensions (viewBox space)
  const width = 800;
  const pad = { top: 20, right: 40, bottom: 40, left: 40 };

  // X-axis range: wider of ±3σ or lo/hi endpoints, whichever is wider
  const sigmaRange = 3;
  const meanXMin = mean - sigmaRange * std;
  const meanXMax = mean + sigmaRange * std;
  const xMin = lo != null ? Math.min(meanXMin, lo) : meanXMin;
  const xMax = hi != null ? Math.max(meanXMax, hi) : meanXMax;

  // Chart body dimensions
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  // Sample points for Gaussian curve
  const samples = 60;
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i <= samples; i++) {
    const x = xMin + (i / samples) * (xMax - xMin);
    const density = gaussianPdf(x, mean, std);
    points.push({ x, y: density });
  }

  // Find max density for Y-normalization (curve height fills chart)
  const maxDensity = Math.max(...points.map((p) => p.y));

  // Position functions
  const toX = (v: number) =>
    pad.left + ((v - xMin) / (xMax - xMin)) * chartW;
  const toY = (density: number) =>
    pad.top + chartH - (density / maxDensity) * chartH;

  // Build curve path
  const curvePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`)
    .join(" ");

  // Build 90% band area (from lo to hi, below curve to X-axis)
  let bandPath = "";
  if (lo != null && hi != null) {
    const bandPoints = points.filter((p) => p.x >= lo && p.x <= hi);
    if (bandPoints.length > 0) {
      const firstX = toX(lo);
      const lastX = toX(hi);
      bandPath = [
        `M ${firstX.toFixed(1)},${(pad.top + chartH).toFixed(1)}`,
        ...bandPoints.map(
          (p) => `L ${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`,
        ),
        `L ${lastX.toFixed(1)},${(pad.top + chartH).toFixed(1)}`,
        "Z",
      ].join(" ");
    }
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {/* 90% credible band — filled area */}
      {bandPath && (
        <path
          d={bandPath}
          fill={color}
          opacity={0.15}
        />
      )}

      {/* Gaussian curve */}
      <path
        d={curvePath}
        stroke={color}
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Mean marker — vertical line */}
      <line
        x1={toX(mean)}
        y1={pad.top}
        x2={toX(mean)}
        y2={pad.top + chartH}
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray="3,3"
      />

      {/* Mean value label */}
      <text
        x={toX(mean)}
        y={pad.top - 6}
        textAnchor="middle"
        fontSize={11}
        fontFamily="var(--f-mono)"
        fill={color}
        fontWeight={600}
      >
        {mean.toFixed(1)}
      </text>

      {/* X-axis line */}
      <line
        x1={pad.left}
        y1={pad.top + chartH}
        x2={pad.left + chartW}
        y2={pad.top + chartH}
        stroke="var(--line-soft)"
        strokeWidth={0.5}
      />

      {/* Lo / Hi labels */}
      {lo != null && (
        <text
          x={toX(lo)}
          y={pad.top + chartH + 16}
          textAnchor="middle"
          fontSize={9}
          fontFamily="var(--f-mono)"
          fill="var(--ink-4)"
        >
          {lo.toFixed(0)}
        </text>
      )}
      {hi != null && (
        <text
          x={toX(hi)}
          y={pad.top + chartH + 16}
          textAnchor="middle"
          fontSize={9}
          fontFamily="var(--f-mono)"
          fill="var(--ink-4)"
        >
          {hi.toFixed(0)}
        </text>
      )}

      {/* Line marker (currently unused; line data pending) */}
      {line != null && (
        <>
          <line
            x1={toX(line)}
            y1={pad.top}
            x2={toX(line)}
            y2={pad.top + chartH}
            stroke="var(--warn)"
            strokeWidth={1.5}
          />
          <text
            x={toX(line)}
            y={pad.top - 6}
            textAnchor="middle"
            fontSize={10}
            fontFamily="var(--f-mono)"
            fill="var(--warn)"
            fontWeight={600}
          >
            Line {line.toFixed(1)}
          </text>
        </>
      )}
    </svg>
  );
}

/**
 * Standard normal PDF for a given x, mean, and std.
 */
function gaussianPdf(x: number, mean: number, std: number): number {
  const z = (x - mean) / std;
  return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
}
