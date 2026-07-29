import {
  describe,
  expect,
  it,
} from "vitest";
import {
  BET_SLIP_SIZING_VERSION,
  DEFAULT_BET_SLIP_SIZING,
  DEFAULT_KELLY_MULTIPLIER,
  parseBetSlipSizingPreference,
  resolveBetSlipBankroll,
  updateBetSlipSizingPreference,
} from "./betSlipSizing";

describe(
  "parseBetSlipSizingPreference",
  () => {
    it(
      "accepts a valid tracked preference",
      () => {
        expect(
          parseBetSlipSizingPreference({
            version:
              BET_SLIP_SIZING_VERSION,
            bankrollMode: "tracked",
            whatIfBankroll: null,
            kellyMultiplier: 0.25,
          }),
        ).toEqual({
          version:
            BET_SLIP_SIZING_VERSION,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: 0.25,
        });
      },
    );

    it(
      "accepts a valid what-if preference",
      () => {
        expect(
          parseBetSlipSizingPreference({
            version:
              BET_SLIP_SIZING_VERSION,
            bankrollMode: "what-if",
            whatIfBankroll: 2500,
            kellyMultiplier: 0.1,
          }),
        ).toEqual({
          version:
            BET_SLIP_SIZING_VERSION,
          bankrollMode: "what-if",
          whatIfBankroll: 2500,
          kellyMultiplier: 0.1,
        });
      },
    );

    it(
      "preserves zero bankroll and multiplier",
      () => {
        expect(
          parseBetSlipSizingPreference({
            version:
              BET_SLIP_SIZING_VERSION,
            bankrollMode: "what-if",
            whatIfBankroll: 0,
            kellyMultiplier: 0,
          }),
        ).toEqual({
          version:
            BET_SLIP_SIZING_VERSION,
          bankrollMode: "what-if",
          whatIfBankroll: 0,
          kellyMultiplier: 0,
        });
      },
    );

    it.each([
      {
        label: "null",
        value: null,
      },
      {
        label: "array",
        value: [],
      },
      {
        label: "empty object",
        value: {},
      },
      {
        label: "wrong version",
        value: {
          version: 2,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: 0.25,
        },
      },
      {
        label: "invalid mode",
        value: {
          version: 1,
          bankrollMode: "invalid",
          whatIfBankroll: null,
          kellyMultiplier: 0.25,
        },
      },
      {
        label: "negative what-if bankroll",
        value: {
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll: -1,
          kellyMultiplier: 0.25,
        },
      },
      {
        label: "negative multiplier",
        value: {
          version: 1,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: -0.01,
        },
      },
      {
        label: "multiplier above one",
        value: {
          version: 1,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier: 1.01,
        },
      },
      {
        label: "NaN multiplier",
        value: {
          version: 1,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier:
            Number.NaN,
        },
      },
      {
        label: "infinite bankroll",
        value: {
          version: 1,
          bankrollMode: "what-if",
          whatIfBankroll:
            Number.POSITIVE_INFINITY,
          kellyMultiplier: 0.25,
        },
      },
    ])(
      "rejects $label",
      ({ value }) => {
        expect(
          parseBetSlipSizingPreference(
            value,
          ),
        ).toBeNull();
      },
    );
  },
);

describe(
  "resolveBetSlipBankroll",
  () => {
    it(
      "uses the tracked bankroll in tracked mode",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "tracked",
              whatIfBankroll: 5000,
            },
            trackedBankroll: 2500,
          }),
        ).toEqual({
          amount: 2500,
          source: "tracked",
        });
      },
    );

    it(
      "does not fall back to the what-if bankroll in tracked mode",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "tracked",
              whatIfBankroll: 5000,
            },
            trackedBankroll: null,
          }),
        ).toEqual({
          amount: null,
          source: "unavailable",
        });
      },
    );

    it(
      "uses the what-if bankroll in what-if mode",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "what-if",
              whatIfBankroll: 1500,
            },
            trackedBankroll: 2500,
          }),
        ).toEqual({
          amount: 1500,
          source: "what-if",
        });
      },
    );

    it(
      "does not fall back to tracked bankroll when what-if value is unavailable",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "what-if",
              whatIfBankroll: null,
            },
            trackedBankroll: 2500,
          }),
        ).toEqual({
          amount: null,
          source: "unavailable",
        });
      },
    );

    it(
      "preserves a zero tracked bankroll",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "tracked",
            },
            trackedBankroll: 0,
          }),
        ).toEqual({
          amount: 0,
          source: "tracked",
        });
      },
    );

    it(
      "preserves a zero what-if bankroll",
      () => {
        expect(
          resolveBetSlipBankroll({
            preference: {
              ...DEFAULT_BET_SLIP_SIZING,
              bankrollMode: "what-if",
              whatIfBankroll: 0,
            },
            trackedBankroll: 2500,
          }),
        ).toEqual({
          amount: 0,
          source: "what-if",
        });
      },
    );

    it.each([
      Number.NaN,
      Number.POSITIVE_INFINITY,
      -1,
    ])(
      "rejects invalid tracked bankroll %s",
      (trackedBankroll) => {
        expect(
          resolveBetSlipBankroll({
            preference:
              DEFAULT_BET_SLIP_SIZING,
            trackedBankroll,
          }),
        ).toEqual({
          amount: null,
          source: "unavailable",
        });
      },
    );
  },
);

describe(
  "updateBetSlipSizingPreference",
  () => {
    it(
      "updates the bankroll mode",
      () => {
        expect(
          updateBetSlipSizingPreference({
            current:
              DEFAULT_BET_SLIP_SIZING,
            update: {
              bankrollMode: "what-if",
            },
          }),
        ).toEqual({
          ...DEFAULT_BET_SLIP_SIZING,
          bankrollMode: "what-if",
        });
      },
    );

    it(
      "updates the what-if bankroll",
      () => {
        expect(
          updateBetSlipSizingPreference({
            current:
              DEFAULT_BET_SLIP_SIZING,
            update: {
              whatIfBankroll: 3000,
            },
          }),
        ).toEqual({
          ...DEFAULT_BET_SLIP_SIZING,
          whatIfBankroll: 3000,
        });
      },
    );

    it(
      "clears the what-if bankroll",
      () => {
        const current = {
          ...DEFAULT_BET_SLIP_SIZING,
          bankrollMode:
            "what-if" as const,
          whatIfBankroll: 3000,
        };

        expect(
          updateBetSlipSizingPreference({
            current,
            update: {
              whatIfBankroll: null,
            },
          }),
        ).toEqual({
          ...current,
          whatIfBankroll: null,
        });
      },
    );

    it(
      "updates the Kelly multiplier",
      () => {
        expect(
          updateBetSlipSizingPreference({
            current:
              DEFAULT_BET_SLIP_SIZING,
            update: {
              kellyMultiplier: 0.1,
            },
          }),
        ).toEqual({
          ...DEFAULT_BET_SLIP_SIZING,
          kellyMultiplier: 0.1,
        });
      },
    );

    it(
      "preserves zero bankroll and multiplier updates",
      () => {
        expect(
          updateBetSlipSizingPreference({
            current:
              DEFAULT_BET_SLIP_SIZING,
            update: {
              bankrollMode: "what-if",
              whatIfBankroll: 0,
              kellyMultiplier: 0,
            },
          }),
        ).toEqual({
          version:
            BET_SLIP_SIZING_VERSION,
          bankrollMode: "what-if",
          whatIfBankroll: 0,
          kellyMultiplier: 0,
        });
      },
    );

    it.each([
      {
        whatIfBankroll: -1,
      },
      {
        whatIfBankroll:
          Number.POSITIVE_INFINITY,
      },
      {
        kellyMultiplier: -0.01,
      },
      {
        kellyMultiplier: 1.01,
      },
      {
        kellyMultiplier:
          Number.NaN,
      },
    ])(
      "preserves the current preference for invalid update %#",
      (update) => {
        expect(
          updateBetSlipSizingPreference({
            current:
              DEFAULT_BET_SLIP_SIZING,
            update,
          }),
        ).toEqual(
          DEFAULT_BET_SLIP_SIZING,
        );
      },
    );
  },
);

describe(
  "default sizing preference",
  () => {
    it(
      "defaults to tracked quarter-Kelly without a what-if bankroll",
      () => {
        expect(
          DEFAULT_BET_SLIP_SIZING,
        ).toEqual({
          version:
            BET_SLIP_SIZING_VERSION,
          bankrollMode: "tracked",
          whatIfBankroll: null,
          kellyMultiplier:
            DEFAULT_KELLY_MULTIPLIER,
        });

        expect(
          DEFAULT_KELLY_MULTIPLIER,
        ).toBe(0.25);
      },
    );
  },
);
