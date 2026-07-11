import { useState } from "react";
import { useDevPanel } from "../../context/DevPanelContext";

/**
 * Floating dev panel. Bottom-right button toggles a small panel of
 * development/verification tools. First tool: Highlight Pending & Blocked.
 *
 * Panel does NOT close on click-outside — suits the audit workflow where
 * you flip highlight on, then click around the app freely while the panel
 * stays open.
 *
 * Deliberately utilitarian styling (mono text, orange accent) to signal
 * this is a dev surface, not a user feature.
 */
export function DevPanel() {
  const { state, setState } = useDevPanel();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        right: 20,
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 8,
      }}
    >
      {/* Panel (expands upward, shown when open) */}
      {isOpen && (
        <div
          style={{
            background: "var(--bg-2)",
            border: "1px solid var(--highlight)",
            borderRadius: 6,
            padding: 14,
            minWidth: 220,
            boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
          }}
        >
          <div
            className="mono upper"
            style={{
              fontSize: 9,
              letterSpacing: "0.12em",
              color: "var(--highlight)",
              marginBottom: 12,
            }}
          >
            Dev Panel
          </div>

          {/* Highlight toggle */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div
              className="mono"
              style={{ fontSize: 11, color: "var(--ink-2)" }}
            >
              Highlight Pending &amp; Blocked
            </div>
            <button
              type="button"
              onClick={() =>
                setState({ highlightPending: !state.highlightPending })
              }
              style={{
                background: state.highlightPending
                  ? "var(--highlight)"
                  : "transparent",
                color: state.highlightPending
                  ? "var(--bg)"
                  : "var(--ink-3)",
                border: `1px solid ${
                  state.highlightPending
                    ? "var(--highlight)"
                    : "var(--line)"
                }`,
                borderRadius: 4,
                padding: "6px 12px",
                fontSize: 11,
                fontWeight: 600,
                fontFamily: "var(--f-mono)",
                cursor: "pointer",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              {state.highlightPending ? "ON" : "OFF"}
            </button>
            <div
              className="mono dim2"
              style={{ fontSize: 9.5, lineHeight: 1.4 }}
            >
              Lights up every pending/blocked field so backend gaps
              are visible during a visual pass.
            </div>
          </div>
        </div>
      )}

      {/* Floating button */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-label="Toggle dev panel"
        style={{
          background: isOpen ? "var(--highlight)" : "var(--bg-2)",
          color: isOpen ? "var(--bg)" : "var(--ink-3)",
          border: `1px solid ${isOpen ? "var(--highlight)" : "var(--line)"}`,
          borderRadius: 6,
          padding: "8px 14px",
          fontSize: 10,
          fontWeight: 700,
          fontFamily: "var(--f-mono)",
          letterSpacing: "0.1em",
          cursor: "pointer",
          boxShadow: "0 2px 12px rgba(0,0,0,0.3)",
        }}
    >
        DEV
        {/* Small indicator dot when highlight is active but panel closed */}
        {!isOpen && state.highlightPending && (
          <span
            style={{
              display: "inline-block",
              width: 6,
              height: 6,
              borderRadius: 3,
              background: "var(--highlight)",
              marginLeft: 6,
              verticalAlign: "middle",
            }}
          />
        )}
      </button>
    </div>
  );
}
