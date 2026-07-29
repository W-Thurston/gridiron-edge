import type { BetLeg } from "../../context/BetSlipContext";
import {
  analyzeBetLeg,
  type BetLegAnalysis,
} from "../../utils/betLegs";
import {
  formatOdds,
} from "../../utils/odds";
import {
  formatStatType,
} from "../../utils/props";
import { PendingChip } from "../field-status/PendingChip";
import { TeamMark } from "../primitives/TeamMark";

type OddsFormat =
  | "american"
  | "decimal";

type BetLegCardProps = {
  leg: BetLeg;
  oddsFormat: OddsFormat;
  bankroll: number | null;
  kellyMultiplier: number;
  onUpdateCurrentOdds: (
    value: number | null,
  ) => void;
  onUpdateProposedStake: (
    value: number | null,
  ) => void;
  onUpdateSportsbook: (
    value: string | null,
  ) => void;
  onUpdateNote: (
    value: string | null,
  ) => void;
  onRemove: () => void;
};

export function BetLegCard({
  leg,
  oddsFormat,
  bankroll,
  kellyMultiplier,
  onUpdateCurrentOdds,
  onUpdateProposedStake,
  onUpdateSportsbook,
  onUpdateNote,
  onRemove,
}: BetLegCardProps) {
  const analysis = analyzeBetLeg({
    leg,
    bankroll,
    kellyMultiplier,
  });

  return (
    <article
      aria-label={legLabel(leg)}
      style={{
        padding: 14,
        backgroundColor: "var(--bg-1)",
        border:
          "1px solid var(--line-soft)",
        borderRadius: 6,
      }}
    >
      <CardHeader
        leg={leg}
        onRemove={onRemove}
      />

      <WagerDescription leg={leg} />

      <PriceSection
        leg={leg}
        analysis={analysis}
        oddsFormat={oddsFormat}
        onUpdateCurrentOdds={
          onUpdateCurrentOdds
        }
      />

      <ModelSection
        leg={leg}
        analysis={analysis}
        oddsFormat={oddsFormat}
        kellyMultiplier={
          kellyMultiplier
        }
      />

      <StakeSection
        leg={leg}
        analysis={analysis}
        onUpdateProposedStake={
          onUpdateProposedStake
        }
      />

      <DraftDetails
        leg={leg}
        onUpdateSportsbook={
          onUpdateSportsbook
        }
        onUpdateNote={onUpdateNote}
      />
    </article>
  );
}

function CardHeader({
  leg,
  onRemove,
}: {
  leg: BetLeg;
  onRemove: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: 12,
        marginBottom: 8,
      }}
    >
      <div
        style={{
          minWidth: 0,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        {leg.kind === "game" ? (
          <>
            <TeamMark
              abbr={leg.awayTeam}
              size={20}
            />

            <span className="dim">@</span>

            <TeamMark
              abbr={leg.homeTeam}
              size={20}
            />
          </>
        ) : (
          <>
            <TeamMark
              abbr={leg.team}
              size={20}
            />

            <div>
              <div
                style={{
                  fontSize: 12,
                  color: "var(--ink)",
                }}
              >
                {leg.playerName}
              </div>

              <div
                className="mono dim2"
                style={{ fontSize: 9 }}
              >
                {leg.position} ·{" "}
                {leg.team}
              </div>
            </div>
          </>
        )}
      </div>

      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${legLabel(
          leg,
        )}`}
        title="Remove leg"
        style={{
          backgroundColor:
            "transparent",
          border: "none",
          padding: "1px 4px",
          fontSize: 16,
          lineHeight: 1,
          color: "var(--ink-3)",
          cursor: "pointer",
          flexShrink: 0,
        }}
      >
        ×
      </button>
    </div>
  );
}

function WagerDescription({
  leg,
}: {
  leg: BetLeg;
}) {
  return (
    <div
      style={{
        marginBottom: 12,
        paddingBottom: 10,
        borderBottom:
          "1px solid var(--line-soft)",
      }}
    >
      <div
        className="mono"
        style={{
          fontSize: 11,
          color: "var(--ink-2)",
          textTransform: "capitalize",
        }}
      >
        {leg.kind === "game"
          ? gameDescription(leg)
          : propDescription(leg)}
      </div>

      <div
        className="mono dim2"
        style={{
          marginTop: 3,
          fontSize: 9,
        }}
      >
        Model:{" "}
        {leg.recommendation.modelKey}
      </div>
    </div>
  );
}

function PriceSection({
  leg,
  analysis,
  oddsFormat,
  onUpdateCurrentOdds,
}: {
  leg: BetLeg;
  analysis: BetLegAnalysis;
  oddsFormat: OddsFormat;
  onUpdateCurrentOdds: (
    value: number | null,
  ) => void;
}) {
  const referenceOdds =
    leg.recommendation
      .referenceAmericanOdds;

  const currentOdds =
    leg.draft.currentAmericanOdds;

  return (
    <section
      aria-label="Price comparison"
      style={{
        display: "grid",
        gridTemplateColumns:
          "repeat(2, minmax(0, 1fr))",
        gap: 10,
        marginBottom: 12,
      }}
    >
      <FieldBlock label="Reference price">
        {referenceOdds == null ? (
          <PendingChip>
            Reference price unavailable
          </PendingChip>
        ) : (
          <span className="mono tnum">
            {formatOdds(
              referenceOdds,
              oddsFormat,
            )}
          </span>
        )}
      </FieldBlock>

      <FieldBlock label="Current price">
        <AmericanOddsInput
          value={currentOdds}
          onChange={
            onUpdateCurrentOdds
          }
        />
      </FieldBlock>

      <FieldBlock label="Model break-even">
        {analysis
          .breakEvenAmericanOdds ==
        null ? (
          <span className="mono dim2">
            Unavailable
          </span>
        ) : (
          <span className="mono tnum">
            {formatOdds(
              analysis
                .breakEvenAmericanOdds,
              oddsFormat,
            )}
          </span>
        )}
      </FieldBlock>

      <FieldBlock label="Price status">
        <PriceStatus
          analysis={analysis}
        />
      </FieldBlock>
    </section>
  );
}

function ModelSection({
  leg,
  analysis,
  oddsFormat,
  kellyMultiplier,
}: {
  leg: BetLeg;
  analysis: BetLegAnalysis;
  oddsFormat: OddsFormat;
  kellyMultiplier: number;
}) {
  const probability =
    leg.recommendation
      .referenceModelProbability;

  return (
    <section
      aria-label="Model analysis"
      style={{
        display: "grid",
        gridTemplateColumns:
          "repeat(2, minmax(0, 1fr))",
        gap: 10,
        marginBottom: 12,
        padding: 10,
        backgroundColor: "var(--bg-2)",
        borderRadius: 5,
      }}
    >
      <Metric
        label="Model probability"
        value={
          probability == null
            ? null
            : `${(
                probability * 100
              ).toFixed(1)}%`
        }
      />

      <Metric
        label="Edge strength"
        value={
          leg.recommendation
            .referenceEdgeStrength
        }
        capitalize
      />

      <Metric
        label="Reference EV"
        value={formatPercent(
          leg.recommendation
            .referenceExpectedValue,
        )}
        tone={evTone(
          leg.recommendation
            .referenceExpectedValue,
        )}
      />

      <Metric
        label="Current EV"
        value={formatPercent(
          analysis.current
            ?.expectedValue ?? null,
        )}
        tone={evTone(
          analysis.current
            ?.expectedValue ?? null,
        )}
      />

      <Metric
        label="Full Kelly"
        value={formatPercent(
          analysis.current
            ?.fullKellyFraction ??
            null,
        )}
      />

      <Metric
        label={`${kellyMultiplier.toFixed(
          2,
        )}× suggested stake`}
        value={formatMoney(
          analysis.suggestedStake,
        )}
      />

      {leg.kind === "prop" && (
        <Metric
          label="Model projection"
          value={
            leg.predictedMean == null
              ? null
              : leg.predictedMean.toFixed(
                  1,
                )
          }
        />
      )}

      <Metric
        label="Current implied"
        value={formatPercent(
          analysis.current
            ?.impliedProbability ??
            null,
        )}
      />

      {analysis.current &&
        leg.draft.currentAmericanOdds !=
          null && (
          <div
            className="mono dim2"
            style={{
              gridColumn: "1 / -1",
              fontSize: 9,
            }}
          >
            Current price{" "}
            {formatOdds(
              leg.draft
                .currentAmericanOdds,
              oddsFormat,
            )}{" "}
            is used for current EV,
            Kelly, payout, and profit.
          </div>
        )}
    </section>
  );
}

function StakeSection({
  leg,
  analysis,
  onUpdateProposedStake,
}: {
  leg: BetLeg;
  analysis: BetLegAnalysis;
  onUpdateProposedStake: (
    value: number | null,
  ) => void;
}) {
  return (
    <section
      aria-label="Stake and payout"
      style={{
        display: "grid",
        gridTemplateColumns:
          "repeat(3, minmax(0, 1fr))",
        gap: 10,
        marginBottom: 12,
      }}
    >
      <FieldBlock label="Your proposed stake">
        <MoneyInput
          value={
            leg.draft.proposedStake
          }
          onChange={
            onUpdateProposedStake
          }
        />
      </FieldBlock>

      <Metric
        label="Potential payout"
        value={formatMoney(
          analysis.payout,
        )}
      />

      <Metric
        label="Potential profit"
        value={formatMoney(
          analysis.profit,
          true,
        )}
        tone={
          analysis.profit == null
            ? "default"
            : analysis.profit >= 0
              ? "positive"
              : "negative"
        }
      />
    </section>
  );
}

function DraftDetails({
  leg,
  onUpdateSportsbook,
  onUpdateNote,
}: {
  leg: BetLeg;
  onUpdateSportsbook: (
    value: string | null,
  ) => void;
  onUpdateNote: (
    value: string | null,
  ) => void;
}) {
  return (
    <details>
      <summary
        className="mono dim"
        style={{
          fontSize: 10,
          cursor: "pointer",
        }}
      >
        Draft details
      </summary>

      <div
        style={{
          display: "grid",
          gap: 8,
          marginTop: 10,
        }}
      >
        <label
          className="upper dim2"
          style={{ fontSize: 9 }}
        >
          Sportsbook
          <input
            type="text"
            value={
              leg.draft.sportsbook ??
              ""
            }
            placeholder="Optional manual entry"
            onChange={(event) =>
              onUpdateSportsbook(
                nullableText(
                  event.target.value,
                ),
              )
            }
            style={textInputStyle}
          />
        </label>

        <label
          className="upper dim2"
          style={{ fontSize: 9 }}
        >
          Note
          <textarea
            value={leg.draft.note ?? ""}
            placeholder="Optional draft note"
            rows={2}
            onChange={(event) =>
              onUpdateNote(
                nullableText(
                  event.target.value,
                ),
              )
            }
            style={{
              ...textInputStyle,
              resize: "vertical",
            }}
          />
        </label>
      </div>
    </details>
  );
}

function AmericanOddsInput({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (
    value: number | null,
  ) => void;
}) {
  return (
    <input
      aria-label="Current American odds"
      type="number"
      step={1}
      value={value ?? ""}
      placeholder="Enter odds"
      onChange={(event) =>
        onChange(
          numberOrNull(
            event.target.value,
          ),
        )
      }
      style={numberInputStyle}
    />
  );
}

function MoneyInput({
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
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      <span className="mono dim">
        $
      </span>

      <input
        aria-label="Proposed stake"
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
        style={numberInputStyle}
      />
    </div>
  );
}

function PriceStatus({
  analysis,
}: {
  analysis: BetLegAnalysis;
}) {
  if (
    analysis.currentPriceIsAcceptable ==
    null
  ) {
    return (
      <span className="mono dim2">
        Threshold unavailable
      </span>
    );
  }

  if (
    analysis.currentPriceIsAcceptable
  ) {
    return (
      <span
        className="mono"
        style={{
          color: "var(--pos)",
        }}
      >
        Positive modeled EV
      </span>
    );
  }

  return (
    <span
      className="mono"
      style={{
        color: "var(--warn)",
      }}
    >
      Below model threshold
    </span>
  );
}

function FieldBlock({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
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

      <div style={{ fontSize: 12 }}>
        {children}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  tone = "default",
  capitalize = false,
}: {
  label: string;
  value: string | null;
  tone?:
    | "default"
    | "positive"
    | "warning"
    | "negative";
  capitalize?: boolean;
}) {
  const color =
    tone === "positive"
      ? "var(--pos)"
      : tone === "warning"
        ? "var(--warn)"
        : tone === "negative"
          ? "var(--neg)"
          : value == null
            ? "var(--ink-4)"
            : "var(--ink-2)";

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
          fontSize: 12,
          color,
          textTransform:
            capitalize
              ? "capitalize"
              : undefined,
        }}
      >
        {value ?? "Unavailable"}
      </div>
    </div>
  );
}

function legLabel(
  leg: BetLeg,
): string {
  return leg.kind === "game"
    ? `${leg.awayTeam} at ${leg.homeTeam} ${leg.market} ${leg.side}`
    : `${leg.playerName} ${formatStatType(
        leg.statType,
      )} ${leg.side}`;
}

function gameDescription(
  leg: Extract<
    BetLeg,
    { kind: "game" }
  >,
): string {
  const line =
    leg.line == null
      ? ""
      : ` · ${formatLine(
          leg.line,
        )}`;

  return `${leg.market} · ${leg.side}${line}`;
}

function propDescription(
  leg: Extract<
    BetLeg,
    { kind: "prop" }
  >,
): string {
  const line =
    leg.line == null
      ? ""
      : ` · ${formatLine(
          leg.line,
        )}`;

  return `${formatStatType(
    leg.statType,
  )} · ${leg.side}${line}`;
}

function formatLine(
  value: number,
): string {
  return value > 0
    ? `+${value}`
    : String(value);
}

function formatPercent(
  value: number | null,
): string | null {
  if (value == null) {
    return null;
  }

  const sign = value > 0 ? "+" : "";

  return `${sign}${(
    value * 100
  ).toFixed(1)}%`;
}

function formatMoney(
  value: number | null,
  includeSign = false,
): string | null {
  if (value == null) {
    return null;
  }

  const sign =
    includeSign && value > 0
      ? "+"
      : "";

  return `${sign}$${value.toFixed(2)}`;
}

function evTone(
  value: number | null,
):
  | "default"
  | "positive"
  | "warning" {
  if (value == null) {
    return "default";
  }

  if (value > 0) {
    return "positive";
  }

  return "warning";
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

function nullableText(
  value: string,
): string | null {
  const normalized = value.trim();

  return normalized === ""
    ? null
    : normalized;
}

const numberInputStyle:
  React.CSSProperties = {
    width: "100%",
    minWidth: 0,
    boxSizing: "border-box",
    padding: "6px 8px",
    backgroundColor: "var(--bg-2)",
    color: "var(--ink)",
    border:
      "1px solid var(--line-soft)",
    borderRadius: 4,
    fontFamily: "var(--f-mono)",
    fontVariantNumeric:
      "tabular-nums",
  };

const textInputStyle:
  React.CSSProperties = {
    display: "block",
    width: "100%",
    boxSizing: "border-box",
    marginTop: 4,
    padding: "7px 8px",
    backgroundColor: "var(--bg-2)",
    color: "var(--ink)",
    border:
      "1px solid var(--line-soft)",
    borderRadius: 4,
    fontFamily: "var(--f-sans)",
    fontSize: 12,
  };
