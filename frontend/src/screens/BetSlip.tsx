import { EdgesTable } from "../components/betslip/EdgesTable";
import { SlipPanel } from "../components/betslip/SlipPanel";
import { useBetSlipSizing } from "../hooks/useBetSlipSizing";

export function BetSlip() {
  const sizing = useBetSlipSizing();

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns:
          "minmax(0, 3fr) minmax(320px, 2fr)",
        gap: 16,
      }}
    >
      <EdgesTable
        bankroll={sizing.bankroll}
        kellyMultiplier={
          sizing.kellyMultiplier
        }
      />

      <SlipPanel sizing={sizing} />
    </div>
  );
}
