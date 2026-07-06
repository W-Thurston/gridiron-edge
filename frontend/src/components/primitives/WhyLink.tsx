import { useState } from "react";
import { useNav } from "../../context/NavContext";

type WhyLinkProps = {
  /** Optional label; defaults to "Why?". Ignored if `dot` is true. */
  label?: string;
  /** Navigation params passed to /explain. Shape is caller-defined. */
  subject?: Record<string, string>;
  /** Compact "?" circle variant instead of labeled chip. */
  dot?: boolean;
  /** Color tone. "info" (blue) default; "pos" (green) for positive contexts. */
  tone?: "info" | "pos";
};

/**
 * Explainability affordance. Small "Why?" chip or "?" dot that navigates
 * to /explain with contextual params. Used across the app anywhere a
 * number could benefit from methodology context.
 *
 * Two modes:
 * - Labeled: `<WhyLink label="Why 71%?" subject={...} />` — chip with text
 * - Dot: `<WhyLink dot subject={...} />` — compact "?" circle for tight spaces
 *
 * Two tones:
 * - "info" (default) — neutral blue
 * - "pos" — green, for positive-context questions like "why is this EV good?"
 *
 * The /explain route is currently a placeholder; navigation still works
 * and lands on the blocked-screen affordance. When /explain is
 * unblocked (W4.5), subject params will drive the explanation content.
 */
export function WhyLink({
  label = "Why?",
  subject = {},
  dot = false,
  tone = "info",
}: WhyLinkProps) {
  const { navigate } = useNav();
  const [hover, setHover] = useState(false);

  const color = tone === "pos" ? "var(--pos)" : "var(--info)";

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigate("/explain", subject);
  };

  if (dot) {
    return (
      <button
        type="button"
        onClick={handleClick}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        aria-label={label}
        title="Why this number?"
        style={{
          width: 15,
          height: 15,
          borderRadius: 8,
          cursor: "pointer",
          padding: 0,
          border: `1px solid color-mix(in oklab, ${color} ${hover ? 80 : 45}%, transparent)`,
          background: hover
            ? `color-mix(in oklab, ${color} 18%, transparent)`
            : "transparent",
          color,
          fontSize: 9.5,
          fontWeight: 700,
          lineHeight: 1,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--f-mono)",
          verticalAlign: "middle",
        }}
      >
        ?
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      aria-label={label}
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        padding: "2px 7px 2px 5px",
        borderRadius: 3,
        cursor: "pointer",
        border: `1px solid color-mix(in oklab, ${color} ${hover ? 55 : 28}%, transparent)`,
        background: `color-mix(in oklab, ${color} ${hover ? 14 : 7}%, transparent)`,
        color,
        fontSize: 9.5,
        fontWeight: 600,
        letterSpacing: "0.04em",
        fontFamily: "var(--f-mono)",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 12,
          height: 12,
          borderRadius: 6,
          background: `color-mix(in oklab, ${color} 22%, transparent)`,
          fontSize: 8,
        }}
      >
        ?
      </span>
      {label}
    </button>
  );
}
