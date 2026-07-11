import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";
import { usePendingHighlight } from "../field-status/usePendingHighlight";

type ComingSoonCardProps = {
  title: string;
  status: FieldStatus | undefined;
};

/**
 * Whole-card placeholder for a scaffolded section. Renders the section
 * title, a field-status badge (pending or blocked), and a "Not yet
 * available" body.
 *
 * When dev panel highlight mode is on, the entire card lights up orange
 * — the card IS the gap indicator.
 *
 * Consolidated from per-screen copies (GameDetail, PlayerProp,
 * TeamsScreen) during W9.8 Substep 2b.
 */
export function ComingSoonCard({ title, status }: ComingSoonCardProps) {
  const highlight = usePendingHighlight();

  return (
    <div
      className="hm-card"
      style={{
        padding: 20,
        ...highlight,
      }}
    >
      <div
        className="upper dim"
        style={{
          fontSize: 10,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>{title}</span>
        {status === "pending" && <PendingField placeholder="" />}
        {status && status !== "pending" && (
          <BlockedField
            blocker={status.blocker}
            roadmap={status.roadmap}
            placeholder=""
          />
        )}
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Not yet available
      </div>
    </div>
  );
}
