import type { components } from "../../api/schema";
import {
  startingQbSourceLabel,
  startingQbStatusMessage,
} from "../field-status/weeklyComponentStatus";

type StartingQbBlock = components["schemas"]["StartingQbBlock"];
type StartingQbSide = components["schemas"]["StartingQbSide"];

/**
 * The quarterbacks the forecast assumed, as persisted with it. A missing
 * record or an unresolved side is stated, never omitted.
 */
export function StartingQuarterbacksCard({
  startingQb,
  awayTeam,
  homeTeam,
}: {
  startingQb: StartingQbBlock;
  awayTeam: string;
  homeTeam: string;
}) {
  const recorded = startingQb.status !== "not_recorded";
  const statusMessage = startingQbStatusMessage(startingQb.status);

  return (
    <div
      className="hm-card"
      style={{ padding: 20 }}
      data-starting-qb-status={startingQb.status}
    >
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
        Assumed Starting QB
      </div>
      {startingQb.status !== "available" && (
        <div
          className="dim"
          role="status"
          style={{ fontSize: 12, marginBottom: 12 }}
        >
          {statusMessage}
        </div>
      )}
      {recorded && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <SideRow team={awayTeam} side={startingQb.away} />
          <SideRow team={homeTeam} side={startingQb.home} />
        </div>
      )}
    </div>
  );
}

function SideRow({
  team,
  side,
}: {
  team: string;
  side: StartingQbSide | null | undefined;
}) {
  if (!side) {
    return (
      <div data-starting-qb-team={team}>
        <strong>{team}</strong>
        <div className="dim" style={{ fontSize: 12 }}>
          Starting quarterback unresolved
        </div>
      </div>
    );
  }

  const label = side.name ?? `${side.player_id} (name unavailable)`;
  return (
    <div data-starting-qb-team={team}>
      <strong>{team}</strong> <span>{label}</span>
      <div className="dim" style={{ fontSize: 12 }}>
        {startingQbSourceLabel(side.source)}
        {side.projection_as_of ? `, snapshot ${side.projection_as_of}` : ""}
      </div>
      {side.differs_from_carry_forward === true && (
        <div style={{ fontSize: 12, color: "var(--warn)" }}>
          Differs from the carried-forward starter
        </div>
      )}
      {side.differs_from_carry_forward == null && (
        <div className="dim" style={{ fontSize: 12 }}>
          Carry-forward comparison unavailable
        </div>
      )}
    </div>
  );
}
