import {
  act,
  renderHook,
  waitFor,
} from "@testing-library/react";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import {
  usePortfolioSummary,
} from "../api/hooks";
import {
  DEFAULT_BET_SLIP_SIZING,
} from "../utils/betSlipSizing";
import {
  useBetSlipSizing,
} from "./useBetSlipSizing";

vi.mock("../api/hooks", () => ({
  usePortfolioSummary: vi.fn(),
}));

const STORAGE_KEY =
  "hm-betslip-sizing-v1";

function mockSummary({
  bankroll = null,
  isLoading = false,
  error = null,
}: {
  bankroll?: number | null;
  isLoading?: boolean;
  error?: unknown;
} = {}) {
  vi.mocked(
    usePortfolioSummary,
  ).mockReturnValue({
    data: {
      bankroll,
    },
    isLoading,
    error,
  } as never);
}

describe("useBetSlipSizing", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockSummary();
  });

  it(
    "defaults to tracked quarter-Kelly sizing",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.preference,
      ).toEqual(
        DEFAULT_BET_SLIP_SIZING,
      );

      expect(
        result.current.bankrollMode,
      ).toBe("tracked");

      expect(
        result.current.kellyMultiplier,
      ).toBe(0.25);

      expect(
        result.current.bankroll,
      ).toBe(2500);

      expect(
        result.current.bankrollSource,
      ).toBe("tracked");
    },
  );

  it(
    "loads a valid persisted preference",
    () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll: 1800,
          kellyMultiplier: 0.1,
        }),
      );

      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.bankrollMode,
      ).toBe("what-if");

      expect(
        result.current.whatIfBankroll,
      ).toBe(1800);

      expect(
        result.current.bankroll,
      ).toBe(1800);

      expect(
        result.current.bankrollSource,
      ).toBe("what-if");

      expect(
        result.current.kellyMultiplier,
      ).toBe(0.1);
    },
  );

  it(
    "falls back to defaults for invalid persisted JSON",
    () => {
      localStorage.setItem(
        STORAGE_KEY,
        "{not-json",
      );

      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.preference,
      ).toEqual(
        DEFAULT_BET_SLIP_SIZING,
      );

      expect(
        result.current.bankroll,
      ).toBe(2500);
    },
  );

  it(
    "falls back to defaults for malformed persisted state",
    () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll: -1,
          kellyMultiplier: 2,
        }),
      );

      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.preference,
      ).toEqual(
        DEFAULT_BET_SLIP_SIZING,
      );

      expect(
        result.current.bankrollSource,
      ).toBe("tracked");
    },
  );

  it(
    "preserves zero as a valid tracked bankroll",
    () => {
      mockSummary({
        bankroll: 0,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.trackedBankroll,
      ).toBe(0);

      expect(
        result.current.bankroll,
      ).toBe(0);

      expect(
        result.current.bankrollSource,
      ).toBe("tracked");
    },
  );

  it(
    "marks tracked bankroll unavailable when absent",
    () => {
      mockSummary({
        bankroll: null,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.trackedBankroll,
      ).toBeNull();

      expect(
        result.current.bankroll,
      ).toBeNull();

      expect(
        result.current.bankrollSource,
      ).toBe("unavailable");
    },
  );

  it(
    "marks tracked bankroll unavailable while loading",
    () => {
      mockSummary({
        bankroll: null,
        isLoading: true,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.isTrackedBankrollLoading,
      ).toBe(true);

      expect(
        result.current.bankroll,
      ).toBeNull();

      expect(
        result.current.bankrollSource,
      ).toBe("unavailable");
    },
  );

  it(
    "exposes tracked-bankroll errors without using a fallback",
    () => {
      const error = new Error(
        "Portfolio unavailable",
      );

      mockSummary({
        bankroll: null,
        error,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      expect(
        result.current.trackedBankrollError,
      ).toBe(error);

      expect(
        result.current.bankroll,
      ).toBeNull();

      expect(
        result.current.bankrollSource,
      ).toBe("unavailable");
    },
  );

  it(
    "switches explicitly to what-if mode",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setWhatIfBankroll(1800);

        result.current
          .setBankrollMode("what-if");
      });

      expect(
        result.current.bankrollMode,
      ).toBe("what-if");

      expect(
        result.current.bankroll,
      ).toBe(1800);

      expect(
        result.current.bankrollSource,
      ).toBe("what-if");
    },
  );

  it(
    "does not fall back to tracked bankroll in incomplete what-if mode",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setBankrollMode("what-if");
      });

      expect(
        result.current.whatIfBankroll,
      ).toBeNull();

      expect(
        result.current.bankroll,
      ).toBeNull();

      expect(
        result.current.bankrollSource,
      ).toBe("unavailable");
    },
  );

  it(
    "switches back to the tracked bankroll",
    () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll: 1800,
          kellyMultiplier: 0.1,
        }),
      );

      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setBankrollMode("tracked");
      });

      expect(
        result.current.bankroll,
      ).toBe(2500);

      expect(
        result.current.bankrollSource,
      ).toBe("tracked");

      expect(
        result.current.whatIfBankroll,
      ).toBe(1800);
    },
  );

  it(
    "updates and preserves a zero what-if bankroll",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setWhatIfBankroll(0);

        result.current
          .setBankrollMode("what-if");
      });

      expect(
        result.current.whatIfBankroll,
      ).toBe(0);

      expect(
        result.current.bankroll,
      ).toBe(0);

      expect(
        result.current.bankrollSource,
      ).toBe("what-if");
    },
  );

  it(
    "updates the Kelly multiplier",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setKellyMultiplier(0.1);
      });

      expect(
        result.current.kellyMultiplier,
      ).toBe(0.1);
    },
  );

  it(
    "rejects invalid sizing updates",
    () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current
          .setWhatIfBankroll(-1);

        result.current
          .setKellyMultiplier(1.5);
      });

      expect(
        result.current.preference,
      ).toEqual(
        DEFAULT_BET_SLIP_SIZING,
      );
    },
  );

  it(
    "persists valid sizing changes",
    async () => {
      mockSummary({
        bankroll: 2500,
      });

      const { result } = renderHook(
        () => useBetSlipSizing(),
      );

      act(() => {
        result.current.updateSizing({
          bankrollMode: "what-if",
          whatIfBankroll: 1800,
          kellyMultiplier: 0.1,
        });
      });

      await waitFor(() => {
        expect(
          JSON.parse(
            localStorage.getItem(
              STORAGE_KEY,
            ) ?? "{}",
          ),
        ).toEqual({
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll: 1800,
          kellyMultiplier: 0.1,
        });
      });
    },
  );
});
