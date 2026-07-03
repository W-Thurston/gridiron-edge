import { useApiHealth } from "../../api/health";

export function OfflineBanner() {
  const { isError } = useApiHealth();

  if (!isError) return null;

  return (
    <div
      role="alert"
      style={{
        background: "var(--neg-dim)",
        color: "var(--bg)",
        padding: "8px 24px",
        fontSize: 12,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
      }}
    >
      <span style={{ fontWeight: 600 }}>Offline —</span>
      <span>
        Can't reach the API. Data on this page may be stale.
      </span>
      <span className="mono" style={{ fontSize: 10, opacity: 0.8 }}>
        Retrying every 30s
      </span>
    </div>
  );
}
