import {
  forwardRef,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";

type WeeklyState = "played" | "projected" | "bye" | "unavailable";
type ActualResult = "W" | "L" | "T";

type TooltipDetails = {
  matchup: string;
  schedule: string;
  outcome: string;
  accessibleLabel: string;
};

type TooltipPosition = {
  left: number;
  top: number;
  width: number;
  placement: "above" | "below";
};

type WinProbabilityCellProps = {
  teamName: string;
  week: number;
  state: WeeklyState;
  opponent?: string | null;
  isHome?: boolean | null;
  gameDate?: string | null;
  gameTime?: string | null;
  winProbability?: number | null;
  actualResult?: ActualResult | null;
  status?: FieldStatus;
  boundary?: boolean;
};

const TOOLTIP_WIDTH = 360;
const TOOLTIP_GAP = 8;
const VIEWPORT_MARGIN = 8;
const APPROXIMATE_TOOLTIP_HEIGHT = 82;

/**
 * Full table cell for a team's weekly chance to win.
 *
 * Scheduled-game cells use a fixed diverging scale centered at 50%.
 * Lower probabilities blend toward the negative token, higher
 * probabilities blend toward the positive token, and 50% is neutral.
 *
 * Matchup details are available through pointer hover and keyboard focus.
 * The tooltip is portaled to the viewport so scroll containers cannot clip it.
 */
export function WinProbabilityCell({
  teamName,
  week,
  state,
  opponent,
  isHome,
  gameDate,
  gameTime,
  winProbability,
  actualResult,
  status,
  boundary = false,
}: WinProbabilityCellProps) {
  const tooltipId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [tooltipPosition, setTooltipPosition] =
    useState<TooltipPosition | null>(null);

  const borderLeftWidth = boundary ? 2 : 1;
  const borderLeftColor = boundary
    ? "var(--line)"
    : "var(--line-soft)";

  const details = buildDetails({
    teamName,
    week,
    state,
    opponent,
    isHome,
    gameDate,
    gameTime,
    winProbability,
    actualResult,
  });

  useLayoutEffect(() => {
    if (!detailsOpen || !triggerRef.current) {
      setTooltipPosition(null);
      return;
    }

    const updatePosition = () => {
      if (!triggerRef.current) return;

      const rect = triggerRef.current.getBoundingClientRect();
      const width = Math.min(
        TOOLTIP_WIDTH,
        window.innerWidth - VIEWPORT_MARGIN * 2,
      );
      const centeredLeft =
        rect.left + rect.width / 2 - width / 2;
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

      setTooltipPosition({
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
  }, [detailsOpen]);

  if (state === "unavailable" || (state !== "bye" && winProbability == null)) {
    return (
      <td
        style={{
          padding: "8px 10px",
          textAlign: "center",
          borderLeftWidth,
          borderLeftStyle: "solid",
          borderLeftColor,
          color: "var(--ink-4)",
          background: "var(--bg-2)",
        }}
      >
        {status === "pending" ? (
          <PendingField />
        ) : status ? (
          <BlockedField blocker={status.blocker} roadmap={status.roadmap} />
        ) : (
          <span
            className="mono tnum"
            aria-label={`${teamName}, Week ${week}: not available`}
          >
            N/A
          </span>
        )}
      </td>
    );
  }

  const isBye = state === "bye";
  const bounded = Math.min(1, Math.max(0, winProbability ?? 0));
  const distance = Math.abs(bounded - 0.5) / 0.5;
  const intensity = 6 + distance * 34;
  const colorToken = bounded < 0.5 ? "var(--neg)" : "var(--pos)";
  const background = isBye
    ? "var(--bg-2)"
    : bounded === 0.5
      ? "var(--bg-2)"
      : `color-mix(in oklab, ${colorToken} ${intensity}%, transparent)`;
  const display = isBye ? "BYE" : formatProbability(winProbability ?? 0);

  return (
    <td
      className="mono tnum"
      style={{
        position: "relative",
        padding: 0,
        textAlign: "center",
        borderLeftWidth,
        borderLeftStyle: "solid",
        borderLeftColor,
        background,
        color: isBye ? "var(--ink-4)" : "var(--ink)",
      }}
    >
      <CellButton
        ref={triggerRef}
        ariaLabel={details?.accessibleLabel ?? `${teamName}, Week ${week}`}
        tooltipId={tooltipId}
        onOpenChange={setDetailsOpen}
      >
        {display}
      </CellButton>

      {detailsOpen && details && tooltipPosition
        ? createPortal(
            <CellTooltip
              id={tooltipId}
              details={details}
              position={tooltipPosition}
            />,
            document.body,
          )
        : null}
    </td>
  );
}

const CellButton = forwardRef<
  HTMLButtonElement,
  {
    ariaLabel: string;
    tooltipId: string;
    children: string;
    onOpenChange: (open: boolean) => void;
  }
>(function CellButton(
  {
    ariaLabel,
    tooltipId,
    children,
    onOpenChange,
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={ariaLabel}
      aria-describedby={tooltipId}
      onMouseEnter={() => onOpenChange(true)}
      onMouseLeave={() => onOpenChange(false)}
      onFocus={() => onOpenChange(true)}
      onBlur={() => onOpenChange(false)}
      style={{
        display: "block",
        width: "100%",
        minWidth: 48,
        height: "100%",
        padding: "9px 7px",
        border: 0,
        backgroundColor: "transparent",
        color: "inherit",
        font: "inherit",
        cursor: "help",
      }}
    >
      {children}
    </button>
  );
});

function CellTooltip({
  id,
  details,
  position,
}: {
  id: string;
  details: TooltipDetails;
  position: TooltipPosition;
}) {
  return (
    <div
      id={id}
      role="tooltip"
      data-placement={position.placement}
      style={{
        position: "fixed",
        zIndex: 1000,
        left: position.left,
        top: position.top,
        transform:
          position.placement === "above"
            ? "translateY(-100%)"
            : undefined,
        width: position.width,
        padding: "9px 11px",
        border: "1px solid var(--line)",
        borderRadius: 4,
        backgroundColor: "var(--bg-1)",
        color: "var(--ink-2)",
        boxShadow: "0 6px 18px rgb(0 0 0 / 35%)",
        pointerEvents: "none",
      }}
    >
      <div
        title={details.matchup}
        style={{
          color: "var(--ink)",
          fontFamily: "var(--f-sans)",
          fontSize: 12,
          fontWeight: 650,
          lineHeight: 1.35,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {details.matchup}
      </div>
      <div
        className="mono dim"
        style={{ marginTop: 3, fontSize: 10, lineHeight: 1.4 }}
      >
        {details.schedule}
      </div>
      <div
        className="mono"
        style={{ marginTop: 3, fontSize: 10, lineHeight: 1.4 }}
      >
        {details.outcome}
      </div>
    </div>
  );
}

function buildDetails({
  teamName,
  week,
  state,
  opponent,
  isHome,
  gameDate,
  gameTime,
  winProbability,
  actualResult,
}: {
  teamName: string;
  week: number;
  state: WeeklyState;
  opponent?: string | null;
  isHome?: boolean | null;
  gameDate?: string | null;
  gameTime?: string | null;
  winProbability?: number | null;
  actualResult?: ActualResult | null;
}): TooltipDetails | null {
  if (state === "unavailable") return null;

  if (state === "bye") {
    return {
      matchup: teamName,
      schedule: `Week ${week}`,
      outcome: "Bye",
      accessibleLabel: `${teamName}, Week ${week}: Bye`,
    };
  }

  const matchup =
    opponent && isHome != null
      ? `${teamName} ${isHome ? "vs." : "at"} ${opponent}`
      : teamName;
  const dateTime = formatDateTime(gameDate, gameTime);
  const schedule = [`Week ${week}`, dateTime]
    .filter(Boolean)
    .join(" · ");

  if (state === "played") {
    const resultText = actualResult
      ? formatActualResult(actualResult)
      : null;
    const outcome = resultText ? `Played · ${resultText}` : "Played";

    return {
      matchup,
      schedule,
      outcome,
      accessibleLabel: [matchup, `Week ${week}`, "Played", resultText]
        .filter(Boolean)
        .join(", "),
    };
  }

  const probabilityText = formatAccessibleProbability(winProbability ?? 0);

  return {
    matchup,
    schedule,
    outcome: `Projected · ${probabilityText} chance to win`,
    accessibleLabel: [
      matchup,
      `Week ${week}`,
      "Projected",
      `${probabilityText} chance to win`,
    ].join(", "),
  };
}

function formatProbability(value: number): string {
  const percentage = value * 100;
  if (value > 0 && percentage < 0.5) return "<1%";
  return `${Math.round(percentage)}%`;
}

function formatAccessibleProbability(value: number): string {
  const percentage = value * 100;
  return percentage === 0 || percentage >= 1
    ? `${percentage.toFixed(1)}%`
    : `${percentage.toFixed(2)}%`;
}

function formatActualResult(result: ActualResult): string {
  if (result === "W") return "Win";
  if (result === "L") return "Loss";
  return "Tie";
}

function formatDateTime(
  gameDate?: string | null,
  gameTime?: string | null,
): string | null {
  const parts: string[] = [];

  if (gameDate) {
    const date = new Date(`${gameDate}T00:00:00Z`);
    if (!Number.isNaN(date.getTime())) {
      parts.push(
        date.toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          year: "numeric",
          timeZone: "UTC",
        }),
      );
    }
  }

  const formattedTime = formatGameTime(gameTime);
  if (formattedTime) parts.push(formattedTime);
  return parts.length > 0 ? parts.join(" · ") : null;
}

function formatGameTime(gameTime?: string | null): string | null {
  if (!gameTime) return null;

  const [hourText, minuteText] = gameTime.split(":");
  const hour = Number(hourText);
  const minute = Number(minuteText);

  if (
    !Number.isInteger(hour) ||
    !Number.isInteger(minute) ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59
  ) {
    return null;
  }

  const period = hour >= 12 ? "PM" : "AM";
  const displayHour = hour % 12 || 12;
  return `${displayHour}:${String(minute).padStart(2, "0")} ${period}`;
}
