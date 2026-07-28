import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";

type HeatCellProps = {
  value: number | null | undefined;
  label: string;
  status?: FieldStatus;
};

/**
 * Renders a normalized probability using a fixed 0–1 heat scale.
 *
 * The numeric value remains the primary encoding; background intensity
 * provides a consistent secondary comparison across probability columns.
 */
export function HeatCell({ value, label, status }: HeatCellProps) {
  if (value == null) {
    if (status === "pending") {
      return <PendingField />;
    }

    if (status) {
      return (
        <BlockedField
          blocker={status.blocker}
          roadmap={status.roadmap}
        />
      );
    }

    return (
      <span
        className="mono tnum"
        title={`${label}: not available`}
        aria-label={`${label}: not available`}
        style={{ color: "var(--ink-4)" }}
      >
        N/A
      </span>
    );
  }

  const bounded = Math.min(1, Math.max(0, value));
  const intensity = 4 + bounded * 30;
  const display = formatProbability(value);

  return (
    <span
      className="mono tnum"
      aria-label={`${label}: ${formatAccessibleProbability(value)}`}
      style={{
        display: "inline-flex",
        minWidth: 44,
        justifyContent: "flex-end",
        padding: "4px 6px",
        borderRadius: 4,
        color: "var(--ink)",
        background: `color-mix(in oklab, var(--pos) ${intensity}%, transparent)`,
      }}
    >
      {display}
    </span>
  );
}

function formatProbability(value: number): string {
  const percentage = value * 100;

  if (value > 0 && percentage < 0.5) {
    return "<1%";
  }

  return `${Math.round(percentage)}%`;
}

function formatAccessibleProbability(value: number): string {
  const percentage = value * 100;

  if (percentage === 0 || percentage >= 1) {
    return `${percentage.toFixed(1)}%`;
  }

  return `${percentage.toFixed(2)}%`;
}
