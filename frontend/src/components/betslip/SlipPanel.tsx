import { useState } from "react";
import { useAppState } from "../../context/AppStateContext";
import { useBetSlip } from "../../context/BetSlipContext";
import type { BetLeg } from "../../context/BetSlipContext";
import type { BetSlipSizingResult } from "../../hooks/useBetSlipSizing";
import {
  americanToDecimal,
  formatOdds,
} from "../../utils/odds";
import { TeamMark } from "../primitives/TeamMark";

export function SlipPanel({
  sizing,
}: {
  sizing: BetSlipSizingResult;
}) {
  const { legs, mode, setMode, remove, clear } = useBetSlip();
  const { state } = useAppState();
  const [stake, setStake] = useState(25);

  const allPriced = legs.every(
    (leg) =>
      leg.draft.currentAmericanOdds !=
      null,
  );

  const combinedDecimal =
    mode === "parlay" &&
    allPriced
      ? legs.reduce(
          (product, leg) =>
            product *
            americanToDecimal(
              leg.draft
                .currentAmericanOdds as number,
            ),
          1,
        )
      : null;

  const totalStake =
    mode === "parlay"
      ? stake
      : stake * legs.length;

  const potentialPayout =
    !allPriced
      ? null
      : mode === "parlay"
        ? stake *
          (combinedDecimal ?? 0)
        : legs.reduce(
            (sum, leg) =>
              sum +
              stake *
                americanToDecimal(
                  leg.draft
                    .currentAmericanOdds as number,
                ),
            0,
          );

  const potentialProfit =
    potentialPayout == null
      ? null
      : potentialPayout -
        totalStake;

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
          {/* Legs */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
            {legs.map((leg) => (
              <LegRow
                key={leg.id}
                leg={leg}
                oddsFormat={state.oddsFormat}
                onRemove={() => remove(leg.id)}
              />
            ))}
          </div>

          {/* Stake input */}
          <div style={{ marginBottom: 16 }}>
            <div className="upper dim2" style={{ fontSize: 9, marginBottom: 6 }}>
              {mode === "parlay" ? "Parlay Stake" : "Stake per Bet"}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span className="mono dim">$</span>
              <input
                type="number"
                value={stake}
                onChange={(e) => setStake(Number(e.target.value) || 0)}
                min={0}
                step={5}
                style={{
                  background: "var(--bg-1)",
                  color: "var(--ink)",
                  border: "1px solid var(--line-soft)",
                  borderRadius: 5,
                  padding: "6px 10px",
                  fontSize: 14,
                  fontFamily: "var(--f-mono)",
                  fontVariantNumeric: "tabular-nums",
                  width: 100,
                }}
              />
            </div>
          </div>

          {/* Totals */}
          <div
            style={{
              padding: 16,
              background: "var(--bg-2)",
              borderRadius: 5,
              marginBottom: 12,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
              }}
            >
              {mode === "parlay" && (
                <MetricLine
                  label="Combined Odds"
                  value={
                    <span className="mono tnum">
                      {combinedDecimal == null
                        ? "Unavailable"
                        : formatOdds(
                            decimalToAmerican(
                              combinedDecimal,
                            ),
                            state.oddsFormat,
                          )}
                    </span>
                  }
                />
              )}
              <MetricLine
                label="Total Stake"
                value={
                  <span className="mono tnum">
                    ${totalStake.toFixed(2)}
                  </span>
                }
              />
              <MetricLine
                label="Potential Payout"
                value={
                  <span
                    className={
                      potentialPayout == null
                        ? "mono tnum dim"
                        : "mono tnum pos"
                    }
                  >
                    {potentialPayout == null
                      ? "Unavailable"
                      : `$${potentialPayout.toFixed(2)}`}
                  </span>
                }
              />
              <MetricLine
                label="Potential Profit"
                value={
                  <span
                    className={
                      potentialProfit == null
                        ? "mono tnum dim"
                        : "mono tnum pos"
                    }
                  >
                    {potentialProfit == null
                      ? "Unavailable"
                      : `${potentialProfit >= 0 ? "+" : ""}$${potentialProfit.toFixed(2)}`}
                  </span>
                }
              />
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={clear}
              style={{
                background: "transparent",
                color: "var(--ink-3)",
                border: "1px solid var(--line-soft)",
                borderRadius: 4,
                padding: "6px 12px",
                fontSize: 12,
                fontFamily: "var(--f-sans)",
                cursor: "pointer",
                flex: 1,
              }}
            >
              Clear Slip
            </button>
          </div>
        </>
      )}
    </div>
  );
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

function LegRow({
  leg,
  oddsFormat,
  onRemove,
}: {
  leg: BetLeg;
  oddsFormat:
    | "american"
    | "decimal";
  onRemove: () => void;
}) {
  const currentAmericanOdds =
    leg.draft.currentAmericanOdds;

  return (
    <div
      style={{
        padding: 10,
        backgroundColor:
          "var(--bg-1)",
        border:
          "1px solid var(--line-soft)",
        borderRadius: 5,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems: "flex-start",
          gap: 8,
        }}
      >
        <div
          style={{
            flex: 1,
            minWidth: 0,
          }}
        >
          {leg.kind === "game" ? (
            <>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  marginBottom: 4,
                  fontSize: 11,
                }}
              >
                <TeamMark
                  abbr={leg.awayTeam}
                />

                <span className="dim">
                  @
                </span>

                <TeamMark
                  abbr={leg.homeTeam}
                />
              </div>

              <div
                className="mono"
                style={{
                  fontSize: 11,
                  color:
                    "var(--ink-2)",
                }}
              >
                {leg.market} ·{" "}
                {leg.side}
                {leg.line != null
                  ? ` · ${leg.line}`
                  : ""}
              </div>
            </>
          ) : (
            <>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  marginBottom: 4,
                  fontSize: 11,
                }}
              >
                <TeamMark
                  abbr={leg.team}
                />

                <span
                  style={{
                    color:
                      "var(--ink-2)",
                  }}
                >
                  {leg.playerName}
                </span>
              </div>

              <div
                className="mono"
                style={{
                  fontSize: 11,
                  color:
                    "var(--ink-2)",
                }}
              >
                {leg.statType} ·{" "}
                {leg.side}
                {leg.line != null
                  ? ` · ${leg.line}`
                  : ""}
              </div>
            </>
          )}
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          {currentAmericanOdds == null ? (
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--warn)",
                whiteSpace: "nowrap",
              }}
            >
              Price unavailable
            </span>
          ) : (
            <span
              className="mono tnum"
              style={{
                fontSize: 12,
                color:
                  "var(--ink-2)",
              }}
            >
              {formatOdds(
                currentAmericanOdds,
                oddsFormat,
              )}
            </span>
          )}

          <button
            type="button"
            onClick={onRemove}
            aria-label="Remove bet leg"
            title="Remove leg"
            style={{
              backgroundColor:
                "transparent",
              border: "none",
              padding: "0 4px",
              cursor: "pointer",
              font: "inherit",
              fontSize: 14,
              color:
                "var(--ink-3)",
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      </div>
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

function MetricLine({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <div className="upper dim2" style={{ fontSize: 9, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 13 }}>{value}</div>
    </div>
  );
}

function decimalToAmerican(decimal: number): number {
  if (decimal >= 2) return Math.round((decimal - 1) * 100);
  return Math.round(-100 / (decimal - 1));
}
