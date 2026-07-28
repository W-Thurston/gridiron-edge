import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";

type HeatCellProps = {
  value: number | null | undefined;
  label: string;
  status?: FieldStatus;
};

/**
 * Table cell for a normalized probability using a fixed 0–1 heat scale.
 *
 * The numeric value remains the primary encoding. The background fills
 * the entire table cell so probability columns form a continuous matrix.
 */
export function HeatCell({
  value,
  label,
  status,
}: HeatCellProps) {
  if (value == null) {
    return (
      <td
        style={{
          padding: "10px 12px",
          textAlign: "right",
          color: "var(--ink-4)",
        }}
      >
        {status === "pending" ? (
          <PendingField />
        ) : status ? (
          <BlockedField
            blocker={status.blocker}
            roadmap={status.roadmap}
          />
        ) : (
          <span
            className="mono tnum"
            title={`${label}: not available`}
            aria-label={`${label}: not available`}
          >
            N/A
          </span>
        )}
      </td>
    );
  }

  const bounded = Math.min(1, Math.max(0, value));
  const intensity = 4 + bounded * 30;
  const display = formatProbability(value);

  return (
    <td
      className="mono tnum"
      aria-label={`${label}: ${formatAccessibleProbability(value)}`}
      style={{
        padding: "10px 12px",
        textAlign: "right",
        color: "var(--ink)",
        background: `color-mix(in oklab, var(--pos) ${intensity}%, transparent)`,
      }}
    >
      {display}
    </td>
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
