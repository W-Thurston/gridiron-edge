import {
  type ReactNode,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

export type ExplainTooltipSection = {
  label: string;
  text: string;
};

type TooltipPosition = {
  left: number;
  top: number;
  width: number;
  placement: "above" | "below";
};

type ExplainTooltipProps = {
  accessibleLabel: string;
  title: string;
  sections: ExplainTooltipSection[];
  children: ReactNode;
  className?: string;
};

const TOOLTIP_WIDTH = 390;
const TOOLTIP_GAP = 8;
const VIEWPORT_MARGIN = 8;
const APPROXIMATE_TOOLTIP_HEIGHT = 220;

/** Accessible hover, focus, and tap explanation portaled outside scroll regions. */
export function ExplainTooltip({
  accessibleLabel,
  title,
  sections,
  children,
  className,
}: ExplainTooltipProps) {
  const tooltipId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) {
      setPosition(null);
      return;
    }

    const updatePosition = () => {
      if (!triggerRef.current) return;
      const rect = triggerRef.current.getBoundingClientRect();
      const width = Math.min(
        TOOLTIP_WIDTH,
        window.innerWidth - VIEWPORT_MARGIN * 2,
      );
      const centeredLeft = rect.left + rect.width / 2 - width / 2;
      const maxLeft = Math.max(
        VIEWPORT_MARGIN,
        window.innerWidth - width - VIEWPORT_MARGIN,
      );
      const left = Math.min(
        Math.max(centeredLeft, VIEWPORT_MARGIN),
        maxLeft,
      );
      const hasRoomAbove =
        rect.top >= APPROXIMATE_TOOLTIP_HEIGHT + TOOLTIP_GAP;
      setPosition({
        left,
        top: hasRoomAbove
          ? rect.top - TOOLTIP_GAP
          : rect.bottom + TOOLTIP_GAP,
        width,
        placement: hasRoomAbove ? "above" : "below",
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open]);

  const closeTransient = () => {
    if (!pinned) setOpen(false);
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={className}
        aria-label={accessibleLabel}
        aria-describedby={open ? tooltipId : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={closeTransient}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          setPinned(false);
          setOpen(false);
        }}
        onClick={() => {
          const nextPinned = !pinned;
          setPinned(nextPinned);
          setOpen(nextPinned);
        }}
      >
        {children}
      </button>
      {open && position
        ? createPortal(
            <div
              id={tooltipId}
              role="tooltip"
              data-placement={position.placement}
              className="explain-tooltip"
              style={{
                left: position.left,
                top: position.top,
                width: position.width,
                transform: position.placement === "above"
                  ? "translateY(-100%)"
                  : undefined,
              }}
            >
              <div className="explain-tooltip__title">{title}</div>
              {sections.map((section) => (
                <section key={section.label} className="explain-tooltip__section">
                  <div className="explain-tooltip__label">{section.label}</div>
                  <div>{section.text}</div>
                </section>
              ))}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
