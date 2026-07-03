import { useState } from "react";
import {
  americanToDecimal,
  impliedProb,
  kelly,
} from "../utils/odds";

export function Tools() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 16,
      }}
    >
      <OddsConverter />
      <KellyCalculator />
      <PayoutCalculator />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Odds converter
// ---------------------------------------------------------------------------

function OddsConverter() {
  const [american, setAmerican] = useState<number | "">(-110);

  const decimal = typeof american === "number" ? americanToDecimal(american) : null;
  const prob = typeof american === "number" ? impliedProb(american) : null;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Odds Converter
      </div>

      <ToolRow label="American Odds (e.g., -110, 200)">
        <input
          type="number"
          value={american}
          onChange={(e) => {
            const v = e.target.value;
            setAmerican(v === "" ? "" : Number(v));
          }}
          style={inputStyle}
        />
      </ToolRow>
      <ToolRow label="Decimal Odds">
        <span className="mono tnum" style={{ fontSize: 14 }}>
          {decimal != null ? decimal.toFixed(3) : "—"}
        </span>
      </ToolRow>
      <ToolRow label="Implied Probability" isLast>
        <span className="mono tnum" style={{ fontSize: 14 }}>
          {prob != null ? `${(prob * 100).toFixed(1)}%` : "—"}
        </span>
      </ToolRow>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Kelly calculator
// ---------------------------------------------------------------------------

function KellyCalculator() {
  const [bankroll, setBankroll] = useState<number | "">(1000);
  const [modelProb, setModelProb] = useState<number | "">(55);
  const [odds, setOdds] = useState<number | "">(-110);

  const p = typeof modelProb === "number" ? modelProb / 100 : null;
  const decimal = typeof odds === "number" ? americanToDecimal(odds) : null;
  const b = decimal != null ? decimal - 1 : null;
  const kellyFrac = p != null && b != null ? kelly(p, b) : null;
  const stake =
    kellyFrac != null && typeof bankroll === "number"
      ? bankroll * kellyFrac
      : null;
  const quarterKelly = stake != null ? stake * 0.25 : null;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Kelly Calculator
      </div>

      <ToolRow label="Bankroll">
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span className="mono dim">$</span>
          <input
            type="number"
            value={bankroll}
            onChange={(e) => {
              const v = e.target.value;
              setBankroll(v === "" ? "" : Number(v));
            }}
            style={inputStyle}
          />
        </div>
      </ToolRow>
      <ToolRow label="Model Win Prob">
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <input
            type="number"
            value={modelProb}
            onChange={(e) => {
              const v = e.target.value;
              setModelProb(v === "" ? "" : Number(v));
            }}
            style={inputStyle}
          />
          <span className="mono dim">%</span>
        </div>
      </ToolRow>
      <ToolRow label="American Odds (e.g., -110, 200)">
        <input
          type="number"
          value={odds}
          onChange={(e) => {
            const v = e.target.value;
            setOdds(v === "" ? "" : Number(v));
          }}
          style={inputStyle}
        />
      </ToolRow>
      <ToolRow label="Full Kelly">
        <span className="mono tnum" style={{ fontSize: 14, color: "var(--pos)" }}>
          {stake != null ? `$${stake.toFixed(2)}` : "—"}
        </span>
      </ToolRow>
      <ToolRow label="Quarter Kelly (0.25x)" isLast>
        <span className="mono tnum" style={{ fontSize: 14, color: "var(--pos)" }}>
          {quarterKelly != null ? `$${quarterKelly.toFixed(2)}` : "—"}
        </span>
      </ToolRow>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Payout calculator
// ---------------------------------------------------------------------------

function PayoutCalculator() {
  const [stake, setStake] = useState<number | "">(100);
  const [odds, setOdds] = useState<number | "">(-110);

  const decimal = typeof odds === "number" ? americanToDecimal(odds) : null;
  const payout =
    typeof stake === "number" && decimal != null ? stake * decimal : null;
  const profit =
    payout != null && typeof stake === "number" ? payout - stake : null;

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Payout Calculator
      </div>

      <ToolRow label="Stake">
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span className="mono dim">$</span>
          <input
            type="number"
            value={stake}
            onChange={(e) => {
              const v = e.target.value;
              setStake(v === "" ? "" : Number(v));
            }}
            style={inputStyle}
          />
        </div>
      </ToolRow>
      <ToolRow label="American Odds (e.g., -110, 200)">
        <input
          type="number"
          value={odds}
          onChange={(e) => {
            const v = e.target.value;
            setOdds(v === "" ? "" : Number(v));
          }}
          style={inputStyle}
        />
      </ToolRow>
      <ToolRow label="Return">
        <span className="mono tnum" style={{ fontSize: 14 }}>
          {payout != null ? `$${payout.toFixed(2)}` : "—"}
        </span>
      </ToolRow>
      <ToolRow label="Profit" isLast>
        <span className="mono tnum pos" style={{ fontSize: 14 }}>
          {profit != null ? `+$${profit.toFixed(2)}` : "—"}
        </span>
      </ToolRow>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

function ToolRow({
  label,
  children,
  isLast,
}: {
  label: string;
  children: React.ReactNode;
  isLast?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 0",
        borderBottom: isLast ? "none" : "1px solid var(--line-soft)",
      }}
    >
      <div style={{ fontSize: 12, color: "var(--ink-2)" }}>{label}</div>
      <div>{children}</div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  background: "var(--bg-1)",
  color: "var(--ink)",
  border: "1px solid var(--line-soft)",
  borderRadius: 5,
  padding: "5px 10px",
  fontSize: 13,
  fontFamily: "var(--f-mono)",
  fontVariantNumeric: "tabular-nums",
  width: 100,
  textAlign: "right",
};
