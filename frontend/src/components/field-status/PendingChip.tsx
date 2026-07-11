import type { ReactNode } from "react";
import { usePendingHighlight } from "./usePendingHighlight";

type PendingChipProps = {
  children: ReactNode;
};

/**
 * Small inline marker for bespoke "X pending" text that isn't backed by
 * a field_status field (e.g., "Confidence pending", "EV pending",
 * "line pending"). Lights up in dev-panel highlight mode alongside the
 * field-status components.
 *
 * Use this instead of hand-written dim text so every "this is pending"
 * moment participates uniformly in the highlight audit.
 */
export function PendingChip({ children }: PendingChipProps) {
  const highlight = usePendingHighlight();
  return (
    <span
      className="mono"
      style={{
        fontSize: 10,
        color: "var(--ink-4)",
        letterSpacing: "0.02em",
        ...highlight,
      }}
    >
      {children}
    </span>
  );
}
