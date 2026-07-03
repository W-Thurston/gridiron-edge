import type { components } from "../../api/schema";
type RatingHistoryPoint = components["schemas"]["RatingHistoryPoint"];


type RatingHistorySparklineProps = {
  history: RatingHistoryPoint[] | null | undefined;
  width?: number;
  height?: number;
};

export function RatingHistorySparkline({
  history,
  width = 300,
  height = 40,
}: RatingHistorySparklineProps) {
  if (!history || history.length === 0) {
    return <span className="dim mono">—</span>;
  }

  const ratings = history.map((p) => p.rating);
  const min = Math.min(...ratings);
  const max = Math.max(...ratings);
  const range = max - min || 1;

  const points = history
    .map((point, i) => {
      const x = (i / (history.length - 1 || 1)) * width;
      const y = height - ((point.rating - min) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");

  const finalRating = ratings[ratings.length - 1];
  const startRating = ratings[0];
  const isUp = finalRating >= startRating;

  return (
    <svg
      width={width}
      height={height}
      style={{ display: "block" }}
    >
      <polyline
        points={points}
        fill="none"
        stroke={isUp ? "var(--pos)" : "var(--neg)"}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
