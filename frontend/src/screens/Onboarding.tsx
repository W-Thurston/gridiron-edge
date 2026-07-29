import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import { useNav } from "../context/NavContext";

export function Onboarding() {
  const { setState } = useAppState();
  const { navigate } = useNav();
  const [step, setStep] = useState(0);
  const [oddsFormat, setOddsFormatLocal] = useState<"american" | "decimal">(
    "american",
  );
  const [bankroll, setBankrollLocal] = useState(1000);

  const handleFinish = () => {
    setState({
      oddsFormat,
      bankroll,
      onboarded: true,
    });
    navigate("/today");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg)",
        padding: 24,
      }}
    >
      <div
        className="hm-card"
        style={{
          padding: 48,
          maxWidth: 560,
          width: "100%",
        }}
      >
        {/* Progress dots */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 8,
            marginBottom: 32,
          }}
        >
          {[0, 1, 2].map((s) => (
            <span
              key={s}
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: s === step ? "var(--pos)" : "var(--bg-3)",
              }}
            />
          ))}
        </div>

        {step === 0 && (
          <WelcomeStep onNext={() => setStep(1)} />
        )}

        {step === 1 && (
          <OddsFormatStep
            value={oddsFormat}
            onChange={setOddsFormatLocal}
            onBack={() => setStep(0)}
            onNext={() => setStep(2)}
          />
        )}

        {step === 2 && (
          <BankrollStep
            value={bankroll}
            onChange={setBankrollLocal}
            onBack={() => setStep(1)}
            onFinish={handleFinish}
          />
        )}
      </div>
    </div>
  );
}

function WelcomeStep({ onNext }: { onNext: () => void }) {
  return (
    <>
      <h1
        className="serif"
        style={{
          fontSize: 32,
          fontWeight: 400,
          margin: 0,
          marginBottom: 12,
          letterSpacing: "-0.01em",
        }}
      >
        Welcome to Gridiron Edge
      </h1>
      <p
        style={{
          fontSize: 14,
          color: "var(--ink-2)",
          lineHeight: 1.6,
          marginBottom: 32,
        }}
      >
        Model-driven NFL analytics with edge detection across game and player
        prop markets. This quick setup gets you configured in three steps.
      </p>
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
        }}
      >
        <PrimaryButton onClick={onNext}>Get started →</PrimaryButton>
      </div>
    </>
  );
}

function OddsFormatStep({
  value,
  onChange,
  onBack,
  onNext,
}: {
  value: "american" | "decimal";
  onChange: (v: "american" | "decimal") => void;
  onBack: () => void;
  onNext: () => void;
}) {
  return (
    <>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
        Step 2 of 3
      </div>
      <h2
        style={{
          fontSize: 22,
          fontWeight: 500,
          margin: 0,
          marginBottom: 12,
        }}
      >
        How do you read odds?
      </h2>
      <p
        style={{
          fontSize: 13,
          color: "var(--ink-2)",
          marginBottom: 32,
        }}
      >
        You can change this anytime in Settings.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 32 }}>
        <OddsChoice
          value="american"
          label="American"
          example="−110 / +150"
          selected={value === "american"}
          onSelect={() => onChange("american")}
        />
        <OddsChoice
          value="decimal"
          label="Decimal"
          example="1.91 / 2.50"
          selected={value === "decimal"}
          onSelect={() => onChange("decimal")}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <SecondaryButton onClick={onBack}>← Back</SecondaryButton>
        <PrimaryButton onClick={onNext}>Next →</PrimaryButton>
      </div>
    </>
  );
}

function OddsChoice({
  label,
  example,
  selected,
  onSelect,
}: {
  value: string;
  label: string;
  example: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      style={{
        width: "100%",
        textAlign: "left",
        background: selected
          ? "color-mix(in oklab, var(--pos) 8%, transparent)"
          : "var(--bg-1)",
        padding: 16,
        borderRadius: 6,
        border: `1px solid ${selected ? "var(--pos)" : "var(--line-soft)"}`,
        cursor: "pointer",
        font: "inherit",
        color: "inherit",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div>
        <div style={{ fontSize: 14, marginBottom: 4 }}>{label}</div>
        <div className="mono tnum dim" style={{ fontSize: 12 }}>
          {example}
        </div>
      </div>
      <div
        style={{
          width: 16,
          height: 16,
          borderRadius: "50%",
          border: `2px solid ${selected ? "var(--pos)" : "var(--line-soft)"}`,
          background: selected ? "var(--pos)" : "transparent",
        }}
      />
    </button>
  );
}

function BankrollStep({
  value,
  onChange,
  onBack,
  onFinish,
}: {
  value: number;
  onChange: (v: number) => void;
  onBack: () => void;
  onFinish: () => void;
}) {
  return (
    <>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
        Step 3 of 3
      </div>
      <h2
        style={{
          fontSize: 22,
          fontWeight: 500,
          margin: 0,
          marginBottom: 12,
        }}
      >
        Calculator bankroll?
      </h2>
      <p
        style={{
          fontSize: 13,
          color: "var(--ink-2)",
          marginBottom: 32,
        }}
      >
        Used as a local what-if value in standalone calculation tools. BetSlip uses tracked portfolio bankroll unless you select a what-if override.
      </p>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 32,
        }}
      >
        <span className="mono" style={{ fontSize: 24, color: "var(--ink-3)" }}>
          $
        </span>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value) || 0)}
          min={0}
          step={100}
          style={{
            background: "var(--bg-1)",
            color: "var(--ink)",
            border: "1px solid var(--line-soft)",
            borderRadius: 6,
            padding: "10px 16px",
            fontSize: 24,
            fontFamily: "var(--f-mono)",
            fontVariantNumeric: "tabular-nums",
            width: 240,
          }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <SecondaryButton onClick={onBack}>← Back</SecondaryButton>
        <PrimaryButton onClick={onFinish}>Finish setup →</PrimaryButton>
      </div>
    </>
  );
}

function PrimaryButton({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "var(--pos)",
        color: "var(--bg)",
        border: "none",
        borderRadius: 5,
        padding: "8px 20px",
        fontSize: 13,
        fontWeight: 600,
        fontFamily: "var(--f-sans)",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}

function SecondaryButton({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "transparent",
        color: "var(--ink-2)",
        border: "1px solid var(--line-soft)",
        borderRadius: 5,
        padding: "8px 20px",
        fontSize: 13,
        fontFamily: "var(--f-sans)",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
