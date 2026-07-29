import { useState } from "react";
import { useAppState } from "../../context/AppStateContext";
import { useBetSlip } from "../../context/BetSlipContext";
import type { BetSlipSizingResult } from "../../hooks/useBetSlipSizing";
import {
  formatOdds,
} from "../../utils/odds";
import {
  summarizeBetSlip,
  type BetSlipSummary,
  type IncompleteBetSlipReason,
} from "../../utils/betSlipSummary";
import { BetLegCard } from "./BetLegCard";


export function SlipPanel({
  sizing,
}: {
  sizing: BetSlipSizingResult;
}) {
  const {
    legs,
    mode,
    setMode,
    updateDraft,
    remove,
    clear,
  } = useBetSlip();

  const { state } = useAppState();

  const [
    parlayStake,
    setParlayStake,
  ] = useState<number | null>(25);

  const summary = summarizeBetSlip({
    legs,
    mode,
    parlayStake,
  });

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Bet Slip
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <ModeButton
            label="Single"
            active={mode === "single"}
            onClick={() => setMode("single")}
          />
          <ModeButton
            label="Parlay"
            active={mode === "parlay"}
            onClick={() => setMode("parlay")}
            disabled={legs.length < 2}
          />
        </div>
      </div>

      <SizingControls sizing={sizing} />

      {legs.length === 0 ? (

        <div
          style={{
            padding: 32,
            textAlign: "center",
            color: "var(--ink-3)",
            fontSize: 13,
          }}
        >
          Your bet slip is empty. Add edges from the table to stage bets.
        </div>
      ) : (
        <>
          {/* Staged wagers */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 12,
              marginBottom: 16,
            }}
          >
            {legs.map((leg) => (
              <BetLegCard
                key={leg.id}
                leg={leg}
                oddsFormat={
                  state.oddsFormat
                }
                bankroll={sizing.bankroll}
                kellyMultiplier={
                  sizing.kellyMultiplier
                }
                onUpdateCurrentOdds={(
                  currentAmericanOdds,
                ) =>
                  updateDraft(leg.id, {
                    currentAmericanOdds,
                  })
                }
                onUpdateProposedStake={(
                  proposedStake,
                ) =>
                  updateDraft(leg.id, {
                    proposedStake,
                  })
                }
                onUpdateSportsbook={(
                  sportsbook,
                ) =>
                  updateDraft(leg.id, {
                    sportsbook,
                  })
                }
                onUpdateNote={(note) =>
                  updateDraft(leg.id, {
                    note,
                  })
                }
                onRemove={() =>
                  remove(leg.id)
                }
              />
            ))}
          </div>

          {/* Aggregate summary */}
          <AggregateSummary
            summary={summary}
            oddsFormat={state.oddsFormat}
            parlayStake={parlayStake}
            onUpdateParlayStake={
              setParlayStake
            }
          />

          {/* Actions */}
          <div>
            <button
              type="button"
              onClick={clear}
              style={{
                width: "100%",
                backgroundColor:
                  "transparent",
                color: "var(--ink-3)",
                border:
                  "1px solid var(--line-soft)",
                borderRadius: 4,
                padding: "7px 12px",
                fontSize: 12,
                fontFamily: "var(--f-sans)",
                cursor: "pointer",
              }}
            >
              Clear Slip
            </button>

            <div
              className="mono dim2"
              style={{
                marginTop: 10,
                fontSize: 9,
                lineHeight: 1.5,
                textAlign: "center",
              }}
            >
              Gridiron Edge provides
              decision-support calculations
              and does not place sportsbook
              wagers.
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function AggregateSummary({
  summary,
  oddsFormat,
  parlayStake,
  onUpdateParlayStake,
}: {
  summary: BetSlipSummary;
  oddsFormat:
    | "american"
    | "decimal";
  parlayStake: number | null;
  onUpdateParlayStake: (
    value: number | null,
  ) => void;
}) {
  return (
    <section
      aria-label="Bet slip summary"
      style={{
        padding: 16,
        marginBottom: 12,
        backgroundColor:
          "var(--bg-2)",
        border:
          "1px solid var(--line-soft)",
        borderRadius: 5,
      }}
    >
      <div
        className="upper dim2"
        style={{
          marginBottom: 10,
          fontSize: 9,
        }}
      >
        {summary.mode === "single"
          ? "Singles Summary"
          : "Parlay Summary"}
      </div>

      {summary.mode === "parlay" && (
        <ParlayStakeInput
          value={parlayStake}
          onChange={
            onUpdateParlayStake
          }
        />
      )}

      {!summary.isComplete && (
        <IncompleteSummary
          reasons={
            summary.incompleteReasons
          }
          mode={summary.mode}
        />
      )}

      {summary.mode === "single" ? (
        <SingleSummaryMetrics
          summary={summary}
        />
      ) : (
        <ParlaySummaryMetrics
          summary={summary}
          oddsFormat={oddsFormat}
        />
      )}
    </section>
  );
}

function SingleSummaryMetrics({
  summary,
}: {
  summary: Extract<
    BetSlipSummary,
    { mode: "single" }
  >;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns:
          "repeat(3, minmax(0, 1fr))",
        gap: 12,
      }}
    >
      <SummaryMetric
        label="Total proposed stake"
        value={formatMoney(
          summary.totalStake,
        )}
      />

      <SummaryMetric
        label="Potential payout"
        value={formatMoney(
          summary.potentialPayout,
        )}
        positive={
          summary.potentialPayout !=
          null
        }
      />

      <SummaryMetric
        label="Potential profit"
        value={formatMoney(
          summary.potentialProfit,
          true,
        )}
        positive={
          summary.potentialProfit !=
            null &&
          summary.potentialProfit >= 0
        }
      />
    </div>
  );
}

function ParlaySummaryMetrics({
  summary,
  oddsFormat,
}: {
  summary: Extract<
    BetSlipSummary,
    { mode: "parlay" }
  >;
  oddsFormat:
    | "american"
    | "decimal";
}) {
  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2, minmax(0, 1fr))",
          gap: 12,
          marginBottom: 12,
        }}
      >
        <SummaryMetric
          label="Combined quoted odds"
          value={
            summary
              .combinedAmericanOdds ==
            null
              ? null
              : formatOdds(
                  summary
                    .combinedAmericanOdds,
                  oddsFormat,
                )
          }
        />

        <SummaryMetric
          label="Parlay stake"
          value={formatMoney(
            summary.parlayStake,
          )}
        />

        <SummaryMetric
          label="Potential payout"
          value={formatMoney(
            summary.potentialPayout,
          )}
          positive={
            summary.potentialPayout !=
            null
          }
        />

        <SummaryMetric
          label="Potential profit"
          value={formatMoney(
            summary.potentialProfit,
            true,
          )}
          positive={
            summary.potentialProfit !=
              null &&
            summary.potentialProfit >= 0
          }
        />
      </div>

      <div
        style={{
          padding: 10,
          backgroundColor:
            "var(--bg-1)",
          borderRadius: 4,
        }}
      >
        <div
          className="mono"
          style={{
            marginBottom: 5,
            fontSize: 10,
            color: "var(--warn)",
          }}
        >
          Parlay correlation is not
          modeled.
        </div>

        <div
          className="mono dim2"
          style={{
            fontSize: 9,
            lineHeight: 1.5,
          }}
        >
          Combined model probability,
          expected value, and Kelly
          sizing are unavailable.
          Quoted payout uses only the
          current prices entered for
          each leg.
        </div>
      </div>
    </>
  );
}

function ParlayStakeInput({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (
    value: number | null,
  ) => void;
}) {
  return (
    <div
      style={{
        marginBottom: 12,
      }}
    >
      <label
        htmlFor="betslip-parlay-stake"
        className="upper dim2"
        style={{
          display: "block",
          marginBottom: 4,
          fontSize: 9,
        }}
      >
        Parlay stake
      </label>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          maxWidth: 160,
        }}
      >
        <span className="mono dim">
          $
        </span>

        <input
          id="betslip-parlay-stake"
          type="number"
          min={0}
          step={5}
          value={value ?? ""}
          placeholder="0.00"
          onChange={(event) =>
            onChange(
              numberOrNull(
                event.target.value,
              ),
            )
          }
          style={{
            width: "100%",
            minWidth: 0,
            boxSizing: "border-box",
            padding: "6px 8px",
            backgroundColor:
              "var(--bg-1)",
            color: "var(--ink)",
            border:
              "1px solid var(--line-soft)",
            borderRadius: 4,
            fontFamily:
              "var(--f-mono)",
            fontVariantNumeric:
              "tabular-nums",
          }}
        />
      </div>
    </div>
  );
}

function IncompleteSummary({
  reasons,
  mode,
}: {
  reasons:
    IncompleteBetSlipReason[];
  mode: "single" | "parlay";
}) {
  if (reasons.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        padding: 10,
        marginBottom: 12,
        backgroundColor:
          "var(--bg-1)",
        borderLeft:
          "3px solid var(--warn)",
        borderRadius: 3,
      }}
    >
      <div
        className="mono"
        style={{
          marginBottom: 5,
          fontSize: 10,
          color: "var(--warn)",
        }}
      >
        Summary incomplete
      </div>

      <ul
        style={{
          margin: 0,
          paddingLeft: 16,
          color: "var(--ink-3)",
          fontSize: 10,
          lineHeight: 1.6,
        }}
      >
        {reasons.map((reason) => (
          <li key={reason}>
            {incompleteReasonLabel(
              reason,
              mode,
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function SummaryMetric({
  label,
  value,
  positive = false,
}: {
  label: string;
  value: string | null;
  positive?: boolean;
}) {
  return (
    <div>
      <div
        className="upper dim2"
        style={{
          marginBottom: 4,
          fontSize: 9,
        }}
      >
        {label}
      </div>

      <div
        className="mono tnum"
        style={{
          fontSize: 13,
          color:
            value == null
              ? "var(--ink-4)"
              : positive
                ? "var(--pos)"
                : "var(--ink-2)",
        }}
      >
        {value ?? "Unavailable"}
      </div>
    </div>
  );
}

function incompleteReasonLabel(
  reason: IncompleteBetSlipReason,
  mode: "single" | "parlay",
): string {
  if (reason === "no_legs") {
    return "Add at least one wager to the slip.";
  }

  if (
    reason ===
    "missing_current_price"
  ) {
    return "Enter current odds for every staged wager.";
  }

  if (
    reason ===
    "missing_proposed_stake"
  ) {
    return mode === "single"
      ? "Enter a proposed stake for every single wager."
      : "Enter all required stake inputs.";
  }

  return "Enter a parlay stake.";
}

function formatMoney(
  value: number | null,
  includePositiveSign = false,
): string | null {
  if (value == null) {
    return null;
  }

  const sign =
    includePositiveSign && value > 0
      ? "+"
      : "";

  return `${sign}$${value.toFixed(2)}`;
}

function numberOrNull(
  value: string,
): number | null {
  if (value.trim() === "") {
    return null;
  }

  const parsed = Number(value);

  return Number.isFinite(parsed)
    ? parsed
    : null;
}

function SizingControls({
  sizing,
}: {
  sizing: BetSlipSizingResult;
}) {
  const sourceLabel =
    sizing.bankrollSource === "tracked"
      ? "Tracked portfolio"
      : sizing.bankrollSource ===
          "what-if"
        ? "What-if"
        : "Unavailable";

  const bankrollLabel =
    sizing.isTrackedBankrollLoading &&
    sizing.bankrollMode === "tracked"
      ? "Loading…"
      : sizing.bankroll == null
        ? "Unavailable"
        : `$${sizing.bankroll.toFixed(2)}`;

  return (
    <div
      style={{
        padding: 12,
        marginBottom: 16,
        backgroundColor: "var(--bg-2)",
        border:
          "1px solid var(--line-soft)",
        borderRadius: 5,
      }}
    >
      <div
        className="upper dim2"
        style={{
          fontSize: 9,
          marginBottom: 8,
        }}
      >
        Sizing Basis
      </div>

      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 10,
        }}
      >
        <ModeButton
          label="Tracked"
          active={
            sizing.bankrollMode ===
            "tracked"
          }
          onClick={() =>
            sizing.setBankrollMode(
              "tracked",
            )
          }
        />

        <ModeButton
          label="What-if"
          active={
            sizing.bankrollMode ===
            "what-if"
          }
          onClick={() =>
            sizing.setBankrollMode(
              "what-if",
            )
          }
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "1fr 1fr",
          gap: 12,
          marginBottom: 10,
        }}
      >
        <div>
          <div
            className="upper dim2"
            style={{
              fontSize: 9,
              marginBottom: 4,
            }}
          >
            Bankroll
          </div>

          <div
            className="mono tnum"
            style={{
              fontSize: 12,
              color:
                sizing.bankroll == null
                  ? "var(--warn)"
                  : "var(--ink-2)",
            }}
          >
            {bankrollLabel}
          </div>
        </div>

        <div>
          <div
            className="upper dim2"
            style={{
              fontSize: 9,
              marginBottom: 4,
            }}
          >
            Source
          </div>

          <div
            className="mono"
            style={{
              fontSize: 12,
              color:
                sizing.bankrollSource ===
                "unavailable"
                  ? "var(--warn)"
                  : "var(--ink-2)",
            }}
          >
            {sourceLabel}
          </div>
        </div>
      </div>

      {sizing.bankrollMode ===
        "what-if" && (
        <div
          style={{
            marginBottom: 10,
          }}
        >
          <label
            htmlFor="betslip-what-if-bankroll"
            className="upper dim2"
            style={{
              display: "block",
              fontSize: 9,
              marginBottom: 4,
            }}
          >
            What-if bankroll
          </label>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            <span className="mono dim">
              $
            </span>

            <input
              id="betslip-what-if-bankroll"
              type="number"
              min={0}
              step={100}
              value={
                sizing.whatIfBankroll ??
                ""
              }
              onChange={(event) => {
                const value =
                  event.target.value;

                sizing.setWhatIfBankroll(
                  value === ""
                    ? null
                    : Number(value),
                );
              }}
              style={{
                width: 120,
                padding: "6px 8px",
                backgroundColor:
                  "var(--bg-1)",
                color: "var(--ink)",
                border:
                  "1px solid var(--line-soft)",
                borderRadius: 4,
                fontFamily:
                  "var(--f-mono)",
                fontVariantNumeric:
                  "tabular-nums",
              }}
            />
          </div>
        </div>
      )}

      <div>
        <label
          htmlFor="betslip-kelly-multiplier"
          className="upper dim2"
          style={{
            display: "block",
            fontSize: 9,
            marginBottom: 4,
          }}
        >
          Kelly multiplier
        </label>

        <select
          id="betslip-kelly-multiplier"
          value={String(
            sizing.kellyMultiplier,
          )}
          onChange={(event) =>
            sizing.setKellyMultiplier(
              Number(
                event.target.value,
              ),
            )
          }
          style={{
            width: "100%",
            padding: "6px 8px",
            backgroundColor:
              "var(--bg-1)",
            color: "var(--ink)",
            border:
              "1px solid var(--line-soft)",
            borderRadius: 4,
            fontFamily:
              "var(--f-mono)",
          }}
        >
          <option value="0.1">
            0.10× Kelly
          </option>

          <option value="0.25">
            0.25× Kelly
          </option>

          <option value="0.5">
            0.50× Kelly
          </option>

          <option value="1">
            1.00× Kelly
          </option>
        </select>
      </div>

      {sizing.trackedBankrollError &&
        sizing.bankrollMode ===
          "tracked" && (
          <div
            className="mono"
            style={{
              marginTop: 8,
              fontSize: 10,
              color: "var(--warn)",
            }}
          >
            Tracked bankroll could not be
            loaded.
          </div>
        )}
    </div>
  );
}

function ModeButton({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: active ? "var(--pos)" : "transparent",
        color: active ? "var(--bg)" : disabled ? "var(--ink-4)" : "var(--ink-2)",
        border: `1px solid ${active ? "var(--pos)" : "var(--line-soft)"}`,
        borderRadius: 4,
        padding: "3px 10px",
        fontSize: 11,
        fontFamily: "var(--f-sans)",
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      {label}
    </button>
  );
}
