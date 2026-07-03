import { EdgesTable } from "../components/betslip/EdgesTable";
import { SlipPanel } from "../components/betslip/SlipPanel";

export function BetSlip() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "3fr 2fr",
        gap: 16,
      }}
    >
      <EdgesTable />
      <SlipPanel />
    </div>
  );
}
