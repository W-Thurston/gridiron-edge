import {
  render,
  screen,
} from "@testing-library/react";
import {
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { BetSlip } from "./BetSlip";
import {
  useBetSlipSizing,
} from "../hooks/useBetSlipSizing";

vi.mock(
  "../hooks/useBetSlipSizing",
  () => ({
    useBetSlipSizing: vi.fn(),
  }),
);

vi.mock(
  "../components/betslip/EdgesTable",
  () => ({
    EdgesTable: ({
      bankroll,
      kellyMultiplier,
    }: {
      bankroll: number | null;
      kellyMultiplier: number;
    }) => (
      <div data-testid="edges-sizing">
        {String(bankroll)}:
        {kellyMultiplier}
      </div>
    ),
  }),
);

vi.mock(
  "../components/betslip/SlipPanel",
  () => ({
    SlipPanel: ({
      sizing,
    }: {
      sizing: {
        bankroll: number | null;
        kellyMultiplier: number;
        bankrollSource: string;
      };
    }) => (
      <div data-testid="panel-sizing">
        {String(sizing.bankroll)}:
        {sizing.kellyMultiplier}:
        {sizing.bankrollSource}
      </div>
    ),
  }),
);

describe("BetSlip", () => {
  it(
    "shares one sizing basis across edges and slip analysis",
    () => {
      vi.mocked(
        useBetSlipSizing,
      ).mockReturnValue({
        preference: {
          version: 1,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: 0.25,
        },
        bankroll: 2500,
        bankrollSource: "tracked",
        trackedBankroll: 2500,
        bankrollMode: "tracked",
        whatIfBankroll: null,
        kellyMultiplier: 0.25,
        isTrackedBankrollLoading:
          false,
        trackedBankrollError: null,
        updateSizing: vi.fn(),
        setBankrollMode: vi.fn(),
        setWhatIfBankroll: vi.fn(),
        setKellyMultiplier: vi.fn(),
      });

      render(<BetSlip />);

      expect(
        screen.getByTestId(
          "edges-sizing",
        ),
      ).toHaveTextContent(
        "2500:0.25",
      );

      expect(
        screen.getByTestId(
          "panel-sizing",
        ),
      ).toHaveTextContent(
        "2500:0.25:tracked",
      );

      expect(
        useBetSlipSizing,
      ).toHaveBeenCalledTimes(1);
    },
  );

  it(
    "passes an unavailable bankroll without substituting zero",
    () => {
      vi.mocked(
        useBetSlipSizing,
      ).mockReturnValue({
        preference: {
          version: 1,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: 0.25,
        },
        bankroll: null,
        bankrollSource:
          "unavailable",
        trackedBankroll: null,
        bankrollMode: "tracked",
        whatIfBankroll: null,
        kellyMultiplier: 0.25,
        isTrackedBankrollLoading:
          false,
        trackedBankrollError:
          new Error(
            "Portfolio unavailable",
          ),
        updateSizing: vi.fn(),
        setBankrollMode: vi.fn(),
        setWhatIfBankroll: vi.fn(),
        setKellyMultiplier: vi.fn(),
      });

      render(<BetSlip />);

      expect(
        screen.getByTestId(
          "edges-sizing",
        ),
      ).toHaveTextContent(
        "null:0.25",
      );

      expect(
        screen.getByTestId(
          "panel-sizing",
        ),
      ).toHaveTextContent(
        "null:0.25:unavailable",
      );
    },
  );
});
