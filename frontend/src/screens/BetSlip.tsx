import { EdgesTable } from "../components/betslip/EdgesTable";
import { SlipPanel } from "../components/betslip/SlipPanel";
import { useBetSlipSizing } from "../hooks/useBetSlipSizing";

export function BetSlip() {
  const sizing = useBetSlipSizing();

  return (
    <div
      className="betslip-layout"
      data-testid="betslip-layout"
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
