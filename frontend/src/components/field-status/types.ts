/**
 * Field status from the API's _meta.field_status envelope.
 * Matches the OpenAPI schema exactly.
 */
export type FieldStatus =
  | "pending"
  | {
      status: "blocked";
      blocker: string;
      roadmap: string;
    };
