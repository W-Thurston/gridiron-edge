type ErrorCardProps = {
  /** The Error instance from a React Query hook. */
  error: Error;
  /** Optional refetch callback. When provided, shows a Retry button. */
  onRetry?: () => void;
  /** Optional message to show above the error details. */
  title?: string;
};

/**
 * Renders a consistent error state across screens.
 * Classifies the error where possible and shows an actionable hint.
 */
export function ErrorCard({ error, onRetry, title = "Something went wrong" }: ErrorCardProps) {
  const errorInfo = classifyError(error);

  return (
    <div
      className="hm-card"
      style={{
        padding: 24,
        borderColor: "var(--neg-dim)",
      }}
    >
      <div
        className="upper"
        style={{
          fontSize: 10,
          marginBottom: 12,
          color: "var(--neg)",
        }}
      >
        {title}
      </div>

      <div style={{ fontSize: 14, color: "var(--ink-2)", marginBottom: 8 }}>
        {errorInfo.message}
      </div>

      {errorInfo.hint && (
        <div
          className="mono"
          style={{ fontSize: 11, color: "var(--ink-3)", marginBottom: 16 }}
        >
          {errorInfo.hint}
        </div>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          style={{
            background: "var(--bg-2)",
            color: "var(--ink)",
            border: "1px solid var(--line-soft)",
            borderRadius: 5,
            padding: "6px 14px",
            fontSize: 12,
            fontWeight: 500,
            fontFamily: "var(--f-sans)",
            cursor: "pointer",
          }}
        >
          Retry
        </button>
      )}
    </div>
  );
}

type ClassifiedError = {
  message: string;
  hint: string | null;
};

function classifyError(error: Error): ClassifiedError {
  const raw = error.message;

  // Try to parse the JSON error body from openapi-fetch responses.
  try {
    const parsed = JSON.parse(raw);
    if (parsed.detail) {
      return {
        message: parsed.detail,
        hint: hintFromDetail(parsed.detail),
      };
    }
  } catch {
    // Not JSON, fall through to raw message handling.
  }

  // Common network patterns.
  if (raw.includes("Failed to fetch") || raw.includes("NetworkError")) {
    return {
      message: "Can't reach the API server.",
      hint: "Is `gridiron api serve` running? Check http://localhost:8000",
    };
  }

  if (raw.includes("timeout")) {
    return {
      message: "Request timed out.",
      hint: "The API may be under load. Try again in a moment.",
    };
  }

  // Fallback.
  return {
    message: raw || "An unexpected error occurred.",
    hint: null,
  };
}

function hintFromDetail(detail: string): string | null {
  if (detail.includes("Prop not found") || detail.includes("Unknown prop_id")) {
    return "This prop is not in the archive.";
  }
  if (detail.includes("Unknown game_id") || detail.includes("Game not found")) {
    return "This game is not in the archive.";
  }
  if (detail.includes("Unknown team")) {
    return "The team abbreviation was not recognized.";
  }
  if (detail.includes("Malformed") || detail.includes("prop_id format")) {
    return "The URL parameter format is invalid.";
  }
  return null;
}
