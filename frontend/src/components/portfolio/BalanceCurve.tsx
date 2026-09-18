type CurvePoint = {
  timestamp?: string;
  bankroll?: number;
};

type BalanceCurveProps = {
  points: CurvePoint[] | null | undefined;
  width?: number;
  height?: number;
};

export function BalanceCurve({
  points,
  width = 600,
  height = 120,
}: BalanceCurveProps) {
  if (!points || points.length === 0) {
    return <span className="dim mono">—</span>;
  }

  const values = points
    .map((point) => point.bankroll)
    .filter((value): value is number => value != null);

  if (values.length === 0) {
    return <span className="dim mono">—</span>;
  }

  const startingBalance = values[0];
  const endingBalance = values[values.length - 1];
  const min = Math.min(...values, startingBalance);
  const max = Math.max(...values, startingBalance);
  const range = max - min || 1;
  const padX = 8;
  const padTop = 8;
  const padBottom = 22;
  const chartWidth = width - padX * 2;
  const chartHeight = height - padTop - padBottom;

  const yForValue = (value: number) =>
    padTop + chartHeight - ((value - min) / range) * chartHeight;

  const line = values
    .map((value, index) => {
      const x = padX + (index / (values.length - 1 || 1)) * chartWidth;
      return `${index === 0 ? "M" : "L"} ${x} ${yForValue(value)}`;
    })
    .join(" ");

  const referenceY = yForValue(startingBalance);
  const isUp = endingBalance >= startingBalance;
  const strokeColor = isUp ? "var(--pos)" : "var(--neg)";
  const fillColor = isUp
    ? "color-mix(in oklab, var(--pos) 15%, transparent)"
    : "color-mix(in oklab, var(--neg) 15%, transparent)";
  const area = `${line} L ${padX + chartWidth} ${padTop + chartHeight} L ${padX} ${padTop + chartHeight} Z`;
  const labelY = Math.min(height - 5, Math.max(11, referenceY + 14));

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={`Balance curve. Starting balance $${startingBalance.toFixed(2)}. Current balance $${endingBalance.toFixed(2)}.`}
      style={{ display: "block", maxWidth: "100%" }}
    >
      <line
        x1={padX}
        x2={padX + chartWidth}
        y1={referenceY}
        y2={referenceY}
        stroke="var(--ink-3)"
        strokeWidth={1}
        strokeDasharray="5 4"
        data-testid="starting-balance-line"
      />
      <text
        x={padX + 4}
        y={labelY}
        fill="var(--ink-3)"
        fontSize={10}
        fontFamily="var(--mono)"
      >
        Starting balance · ${startingBalance.toFixed(2)}
      </text>
      <path d={area} fill={fillColor} stroke="none" />
      <path
        d={line}
        fill="none"
        stroke={strokeColor}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
