import type { components } from "../../api/schema";
import { getEdgeResultPresentation } from "./edgeResultStatus";

type EdgeDiagnostics =
  components["schemas"]["EdgeDiagnosticsResponse"];

type EdgeResultStatusProps = {
  diagnostics: EdgeDiagnostics;
  compact?: boolean;
};

export function EdgeResultStatus({
  diagnostics,
  compact = false,
}: EdgeResultStatusProps) {
  const presentation = getEdgeResultPresentation(diagnostics);
  if (!presentation) return null;

  const color =
    presentation.kind === "blocked"
      ? "var(--warn)"
      : presentation.kind === "filtered"
        ? "var(--info)"
        : "var(--ink-3)";

  return (
    <div
      role="status"
      data-edge-result-kind={presentation.kind}
      style={{
        padding: compact ? 0 : 24,
        textAlign: compact ? "left" : "center",
      }}
    >
      <div
        className="mono"
        style={{
          color,
          fontSize: compact ? 11 : 12,
          fontWeight: 600,
          marginBottom: 6,
        }}
      >
        {presentation.title}
      </div>
      <div
        className="mono dim2"
        style={{
          fontSize: compact ? 10 : 11,
          lineHeight: 1.5,
        }}
      >
        {presentation.detail}
      </div>
      {presentation.blockerMessages.length > 1 && (
        <ul
          style={{
            margin: "10px 0 0",
            paddingLeft: compact ? 18 : 0,
            listStylePosition: compact ? "outside" : "inside",
          }}
        >
          {presentation.blockerMessages.map((message) => (
            <li
              key={message}
              className="mono dim2"
              style={{ fontSize: 10, marginTop: 4 }}
            >
              {message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
