type CurvePoint = {
  timestamp?: string;
  bankroll?: number;
};

type BalanceCurveProps = {
  points: CurvePoint[] | null | undefined;
  width?: number;
  height?: number;
};

export function BalanceCurve({ points, width = 600, height = 120 }: BalanceCurveProps) {
  if (!points || points.length === 0) {
    return <span className="dim mono">—</span>;
  }

  const values = points
    .map((p) => p.bankroll)
    .filter((v): v is number => v != null);

  if (values.length === 0) {
    return <span className="dim mono">—</span>;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const start = values[0];
  const end = values[values.length - 1];

  // Padding so the line isn't right against the edges.
  const pad = 8;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;

  const line = values
    .map((v, i) => {
      const x = pad + (i / (values.length - 1 || 1)) * chartW;
      const y = pad + chartH - ((v - min) / range) * chartH;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const isUp = end >= start;
  const strokeColor = isUp ? "var(--pos)" : "var(--neg)";
  const fillColor = isUp
    ? "color-mix(in oklab, var(--pos) 15%, transparent)"
    : "color-mix(in oklab, var(--neg) 15%, transparent)";

  // Area path — same line + close down to bottom.
  const areaEnd = `L ${pad + chartW} ${pad + chartH} L ${pad} ${pad + chartH} Z`;
  const area = line + " " + areaEnd;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={area} fill={fillColor} stroke="none" />
      <path d={line} fill="none" stroke={strokeColor} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
