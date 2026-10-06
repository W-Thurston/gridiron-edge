import type { components } from "../../api/schema";
import { recapFamilyLabel } from "../field-status/recapStatus";
import type { RecapFamily } from "../field-status/recapStatus";

type WinComparison = components["schemas"]["RecapWinComparison"];
type PointComparison = components["schemas"]["RecapPointComparison"];
type ProbabilityMetrics = components["schemas"]["RecapProbabilityMetrics"];
type ErrorMetrics = components["schemas"]["RecapErrorMetrics"];

type Row<T> = [string, (side: T) => string];

const WIN_ROWS: Row<ProbabilityMetrics>[] = [
  ["Log loss", (side) => fixed(side.log_loss, 4)],
  ["Accuracy", (side) => percent(side.accuracy)],
  ["Games", (side) => String(side.count)],
];

function pointRows(family: Exclude<RecapFamily, "win">): Row<ErrorMetrics>[] {
  return [
    ["RMSE", (side) => fixed(side.rmse, 2)],
    ["Bias", (side) => signed(side.bias, 2)],
    [family === "projected_score" ? "Team scores" : "Games", (side) => String(side.count)],
  ];
}

const HEADLINE: Record<RecapFamily, string> = {
  win: "Brier",
  spread: "MAE, points",
  total: "MAE, points",
  projected_score: "MAE, points per team",
};

/** Model against the closing market for one family: headline metric, who was closer, secondary metrics. */
export function FamilyScoreCard(
  props:
    | { family: "win"; comparison: WinComparison }
    | { family: Exclude<RecapFamily, "win">; comparison: PointComparison },
) {
  const { family } = props;
  if (props.family === "win") {
    const { model, market, closer } = props.comparison;
    return (
      <Card family={family} model={model.brier} market={market.brier} digits={4} closer={closer}
        rows={WIN_ROWS} modelSide={model} marketSide={market} />
    );
  }
  const { model, market, closer } = props.comparison;
  return (
    <Card family={family} model={model.mae} market={market.mae} digits={2} closer={closer}
      rows={pointRows(props.family)} modelSide={model} marketSide={market} />
  );
}

function Card<T>({
  family, model, market, digits, closer, rows, modelSide, marketSide,
}: {
  family: RecapFamily;
  model: number | null | undefined;
  market: number | null | undefined;
  digits: number;
  closer: components["schemas"]["RecapCloserCounts"];
  rows: Row<T>[];
  modelSide: T;
  marketSide: T;
}) {
  const compared = model != null && market != null;
  const verdict = !compared ? null : model < market ? "model" : market < model ? "market" : "even";
  const label = recapFamilyLabel(family);
  return (
    <article className="hm-card pf-fam" data-recap-family={family} aria-label={`${label}: model against the closing market`}>
      <div className="pf-fam-top">
        <span className="pf-fam-title">{label}</span>
        {verdict && (
          <span className={`pf-verdict pf-verdict-${verdict}`} data-verdict={verdict}>
            {verdict === "model" ? "Model ahead" : verdict === "market" ? "Market ahead" : "Even"}
          </span>
        )}
      </div>
      <div className="pf-vs">
        <div>
          <span className="pf-vs-label">Model</span>
          <span className={`pf-vs-num${verdict === "model" ? " pf-better" : ""}`}>{fixed(model, digits)}</span>
        </div>
        <div>
          <span className="pf-vs-label">Closing market</span>
          <span className={`pf-vs-num${verdict === "model" ? " pf-worse" : ""}`}>{fixed(market, digits)}</span>
        </div>
      </div>
      <div className="pf-headline-label">{HEADLINE[family]} · lower is better</div>
      <CloserSplit closer={closer} />
      <table className="pf-minor">
        <thead>
          <tr>
            <th scope="col"><span className="visually-hidden">Metric</span></th>
            <th scope="col">Model</th>
            <th scope="col">Market</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, value]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td>{value(modelSide)}</td>
              <td>{value(marketSide)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

/** Games where the model, neither, or the market was closer to the result. */
export function CloserSplit({ closer }: { closer: components["schemas"]["RecapCloserCounts"] }) {
  const total = closer.model + closer.tie + closer.market;
  return (
    <div className="pf-split">
      {total > 0 && (
        <div className="pf-split-bar" aria-hidden="true">
          <i style={{ flex: closer.model, background: "var(--pos)" }} />
          <i style={{ flex: closer.tie, background: "var(--line)" }} />
          <i style={{ flex: closer.market, background: "var(--ink-3)" }} />
        </div>
      )}
      <div className="pf-split-legend" aria-label="Closer to the result">
        <span style={{ color: "var(--pos)" }}>Model closer {closer.model}</span>
        <span>Tie {closer.tie}</span>
        <span>Market {closer.market}</span>
      </div>
    </div>
  );
}

function fixed(value: number | null | undefined, digits: number) {
  return value == null ? "—" : value.toFixed(digits);
}
function percent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
function signed(value: number | null | undefined, digits: number) {
  return value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}
