import type { ReactNode } from "react";
import { BlockedField } from "./BlockedField";
import { PendingField } from "./PendingField";
import type { FieldStatus } from "./types";

type FieldValueProps = {
  /** The actual value from the API. If null/undefined, we defer to status. */
  value: ReactNode | null | undefined;
  /** The field_status entry from _meta.field_status for this field. */
  status?: FieldStatus;
  /** Placeholder text when rendering pending/blocked states. Defaults to em dash. */
  placeholder?: string;
};

/**
 * Renders a value with awareness of its field_status.
 *
 * - Populated value: renders the value.
 * - Null/undefined value with no status: renders em dash.
 * - Null/undefined value with status "pending": renders <PendingField />.
 * - Null/undefined value with status BlockedStatus: renders <BlockedField />.
 *
 * When both a value and a status are present, the value wins — status
 * is metadata about *why* a null exists, not an override.
 */
export function FieldValue({
  value,
  status,
  placeholder = "—",
}: FieldValueProps) {
  const hasValue = value !== null && value !== undefined && value !== "";

  if (hasValue) {
    return <>{value}</>;
  }

  if (!status) {
    return <span className="mono tnum dim2">{placeholder}</span>;
  }

  if (status === "pending") {
    return <PendingField placeholder={placeholder} />;
  }

  return (
    <BlockedField
      blocker={status.blocker}
      roadmap={status.roadmap}
      placeholder={placeholder}
    />
  );
}
