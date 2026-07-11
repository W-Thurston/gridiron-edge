import type { CSSProperties } from "react";
import { useDevPanel } from "../../context/DevPanelContext";

/**
 * Returns highlight styles when the dev panel's "Highlight Pending &
 * Blocked" mode is on; an empty object otherwise.
 *
 * Spread the result into a component's inline style to make it light up
 * during a visual audit pass:
 *
 *   const highlight = usePendingHighlight();
 *   <span style={{ ...baseStyle, ...highlight }}>
 *
 * When highlight mode is off, returns {} so components render exactly as
 * they do normally — zero visual change.
 */
export function usePendingHighlight(): CSSProperties {
  const { state } = useDevPanel();

  if (!state.highlightPending) {
    return {};
  }

  return {
    outline: "2px solid var(--highlight)",
    outlineOffset: "1px",
    background: "color-mix(in oklab, var(--highlight) 15%, transparent)",
    borderRadius: "3px",
  };
}
