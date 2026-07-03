import { useAppState } from "../context/AppStateContext";
import { useBetSlip } from "../context/BetSlipContext";
import { useNav } from "../context/NavContext";

export function Settings() {
  const { state, setState } = useAppState();
  const { clear: clearBetSlip } = useBetSlip();
  const { navigate } = useNav();

  const handleReset = () => {
    if (confirm("Reset all app state? This clears bankroll, bet slip, odds format, and onboarding.")) {
      localStorage.clear();
      sessionStorage.clear();
      // Force reload so all providers re-initialize from defaults.
      window.location.reload();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Settings
        </div>

        <SettingsRow
          label="Odds Format"
          description="How odds are displayed across the app."
          control={
            <ToggleGroup
              value={state.oddsFormat}
              options={[
                { value: "american", label: "American" },
                { value: "decimal", label: "Decimal" },
              ]}
              onChange={(v) => setState({ oddsFormat: v })}
            />
          }
        />

        <SettingsRow
          label="Bankroll"
          description="Current bankroll used for Kelly stake sizing."
          control={
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="mono dim">$</span>
              <input
                type="number"
                value={state.bankroll}
                onChange={(e) =>
                  setState({ bankroll: Number(e.target.value) || 0 })
                }
                min={0}
                step={100}
                style={numberInputStyle}
              />
            </div>
          }
        />

        <SettingsRow
          label="Alerts"
          description="Notification badge count."
          control={
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span className="mono tnum" style={{ color: "var(--ink-2)" }}>
                {state.alerts}
              </span>
              <button
                onClick={() => setState({ alerts: 0 })}
                style={secondaryButtonStyle}
              >
                Clear
              </button>
            </div>
          }
        />

        <SettingsRow
          label="Onboarded"
          description="First-run flow completed."
          control={
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span
                className="mono"
                style={{
                  color: state.onboarded ? "var(--pos)" : "var(--warn)",
                }}
              >
                {state.onboarded ? "Yes" : "No"}
              </span>
              <button
                onClick={() => {
                  setState({ onboarded: false });
                  navigate("/onboarding");
                }}
                style={secondaryButtonStyle}
              >
                Redo Onboarding
              </button>
            </div>
          }
        />

        <SettingsRow
          label="Bet Slip"
          description="Current bet slip contents."
          control={
            <button onClick={clearBetSlip} style={secondaryButtonStyle}>
              Clear Slip
            </button>
          }
          isLast
        />
      </div>

      {/* Danger zone */}
      <div
        className="hm-card"
        style={{
          padding: 24,
          borderColor: "var(--neg-dim)",
        }}
      >
        <div
          className="upper"
          style={{ fontSize: 10, marginBottom: 16, color: "var(--neg)" }}
        >
          Danger Zone
        </div>

        <SettingsRow
          label="Reset App State"
          description="Clears all localStorage and sessionStorage. Bankroll, bet slip, odds format, and onboarding all return to defaults."
          control={
            <button onClick={handleReset} style={dangerButtonStyle}>
              Reset Everything
            </button>
          }
          isLast
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SettingsRow({
  label,
  description,
  control,
  isLast,
}: {
  label: string;
  description: string;
  control: React.ReactNode;
  isLast?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 24,
        padding: "16px 0",
        borderBottom: isLast ? "none" : "1px solid var(--line-soft)",
      }}
    >
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 14, marginBottom: 4 }}>{label}</div>
        <div className="dim" style={{ fontSize: 12 }}>
          {description}
        </div>
      </div>
      <div>{control}</div>
    </div>
  );
}

function ToggleGroup<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            background: opt.value === value ? "var(--pos)" : "transparent",
            color: opt.value === value ? "var(--bg)" : "var(--ink-2)",
            border: `1px solid ${
              opt.value === value ? "var(--pos)" : "var(--line-soft)"
            }`,
            borderRadius: 4,
            padding: "5px 12px",
            fontSize: 12,
            fontFamily: "var(--f-sans)",
            cursor: "pointer",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

const numberInputStyle: React.CSSProperties = {
  background: "var(--bg-1)",
  color: "var(--ink)",
  border: "1px solid var(--line-soft)",
  borderRadius: 5,
  padding: "5px 10px",
  fontSize: 13,
  fontFamily: "var(--f-mono)",
  fontVariantNumeric: "tabular-nums",
  width: 120,
};

const secondaryButtonStyle: React.CSSProperties = {
  background: "transparent",
  color: "var(--ink-2)",
  border: "1px solid var(--line-soft)",
  borderRadius: 4,
  padding: "5px 12px",
  fontSize: 12,
  fontFamily: "var(--f-sans)",
  cursor: "pointer",
};

const dangerButtonStyle: React.CSSProperties = {
  background: "var(--neg)",
  color: "var(--bg)",
  border: "none",
  borderRadius: 4,
  padding: "6px 16px",
  fontSize: 12,
  fontWeight: 600,
  fontFamily: "var(--f-sans)",
  cursor: "pointer",
};
