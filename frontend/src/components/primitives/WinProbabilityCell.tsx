import { useId, useState } from "react";
import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";

type WeeklyState = "played" | "projected" | "bye" | "unavailable";
type ActualResult = "W" | "L" | "T";

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

/**
 * Full table cell for a team's weekly chance to win.
 *
 * Scheduled-game cells use a fixed diverging scale centered at 50%.
 * Lower probabilities blend toward the negative token, higher
 * probabilities blend toward the positive token, and 50% is neutral.
 *
 * Matchup details are available through both pointer hover and keyboard
 * focus. Bye and unavailable states remain distinct from a 0% chance.
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
  const [detailsOpen, setDetailsOpen] = useState(false);

  const borderLeftWidth = boundary ? 2 : 1;
  const borderLeftColor = boundary
    ? "var(--line)"
    : "var(--line-soft)";

  if (state === "bye") {
    const details = `${teamName}, Week ${week}: Bye`;

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
          background: "var(--bg-2)",
          color: "var(--ink-4)",
        }}
      >
        <CellButton
          ariaLabel={details}
          tooltipId={tooltipId}
          onOpenChange={setDetailsOpen}
        >
          BYE
        </CellButton>
        {detailsOpen && <CellTooltip id={tooltipId}>{details}</CellTooltip>}
      </td>
    );
  }

  if (state === "unavailable" || winProbability == null) {
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

  const bounded = Math.min(1, Math.max(0, winProbability));
  const distance = Math.abs(bounded - 0.5) / 0.5;
  const intensity = 6 + distance * 34;
  const colorToken = bounded < 0.5 ? "var(--neg)" : "var(--pos)";
  const background =
    bounded === 0.5
      ? "var(--bg-2)"
      : `color-mix(in oklab, ${colorToken} ${intensity}%, transparent)`;

  const display = formatProbability(winProbability);
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
        color: "var(--ink)",
      }}
    >
      <CellButton
        ariaLabel={details}
        tooltipId={tooltipId}
        onOpenChange={setDetailsOpen}
      >
        {display}
      </CellButton>
      {detailsOpen && <CellTooltip id={tooltipId}>{details}</CellTooltip>}
    </td>
  );
}

function CellButton({
  ariaLabel,
  tooltipId,
  children,
  onOpenChange,
}: {
  ariaLabel: string;
  tooltipId: string;
  children: string;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <button
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
        background: "transparent",
        color: "inherit",
        font: "inherit",
        cursor: "help",
      }}
    >
      {children}
    </button>
  );
}

function CellTooltip({ id, children }: { id: string; children: string }) {
  return (
    <span
      id={id}
      role="tooltip"
      className="mono"
      style={{
        position: "absolute",
        zIndex: 20,
        left: "50%",
        bottom: "calc(100% + 6px)",
        transform: "translateX(-50%)",
        width: "max-content",
        maxWidth: 260,
        padding: "7px 9px",
        border: "1px solid var(--line)",
        borderRadius: 4,
        background: "var(--bg)",
        color: "var(--ink-2)",
        boxShadow: "0 6px 18px rgb(0 0 0 / 35%)",
        fontSize: 10,
        lineHeight: 1.45,
        textAlign: "left",
        whiteSpace: "normal",
        pointerEvents: "none",
      }}
    >
      {children}
    </span>
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
  state: "played" | "projected";
  opponent?: string | null;
  isHome?: boolean | null;
  gameDate?: string | null;
  gameTime?: string | null;
  winProbability: number;
  actualResult?: ActualResult | null;
}): string {
  const matchup =
    opponent && isHome != null
      ? `${teamName} ${isHome ? "vs." : "at"} ${opponent}`
      : teamName;

  const parts = [matchup, `Week ${week}`];
  const dateTime = formatDateTime(gameDate, gameTime);

  if (dateTime) {
    parts.push(dateTime);
  }

  if (state === "played") {
    parts.push(
      actualResult ? `Played — ${formatActualResult(actualResult)}` : "Played",
    );
  } else {
    parts.push(`${formatAccessibleProbability(winProbability)} chance to win`);
    parts.push("Projected");
  }

  return parts.join(" · ");
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

function formatActualResult(result: ActualResult): string {
  if (result === "W") {
    return "win";
  }

  if (result === "L") {
    return "loss";
  }

  return "tie";
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

  if (formattedTime) {
    parts.push(formattedTime);
  }

  return parts.length > 0 ? parts.join(" · ") : null;
}

function formatGameTime(gameTime?: string | null): string | null {
  if (!gameTime) {
    return null;
  }

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
