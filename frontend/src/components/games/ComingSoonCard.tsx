import { BlockedField } from "../field-status/BlockedField";
import { PendingField } from "../field-status/PendingField";
import type { FieldStatus } from "../field-status/types";

type ComingSoonCardProps = {
  title: string;
  status: FieldStatus | undefined;
};

/**
 * Whole-card placeholder for a scaffolded GameDetail section.
 * Renders a card with the section title and a field-status indicator
 * showing why the card is empty.
 */
export function ComingSoonCard({ title, status }: ComingSoonCardProps) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
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
