import type { ReactNode } from "react";
import type { components } from "../../api/schema";
import { recapEdgeStatusMessage, recapEdgeStatusShort } from "../field-status/recapStatus";
import { money, record, roi, signedMoney } from "../recap/recapFormat";
import { CompareChart } from "./CompareChart";
import { CountsStrip } from "./CountsStrip";
import { FamilyScoreCard } from "./FamilyScoreCard";
import { MethodDetails } from "./MethodDetails";
import { SectionHead } from "./SectionHead";
import { SegmentedControl } from "./SegmentedControl";
import { MODEL_STYLES, SEASON_METRICS, seasonComparison, seasonWeekMetrics } from "./compareData";
import type { CompareModelId, SeasonGames, SeasonView } from "./compareData";
import { fixed, formatTimestamp, percent } from "./format";

type Season = components["schemas"]["SeasonRecapResponse"];
type SeasonWeek = components["schemas"]["SeasonRecapWeek"];
type Scope = components["schemas"]["SeasonRecapScope"];
type WeekMetrics = components["schemas"]["RecapWeekMetrics"];
type Paired = components["schemas"]["RecapPairedDifference"];
type SideCalibration = components["schemas"]["RecapSideCalibration"];
type WagerSummary = components["schemas"]["RecapWagerSummary"];
type CloserCounts = components["schemas"]["RecapCloserCounts"];

/** The model is the emphasized series (the champion's color), the closing market the neutral reference. */
const MODEL = MODEL_STYLES.champion.color;
const MARKET = MODEL_STYLES.closing_line.color;

export type SeasonState = {
  games: SeasonGames;
  view: SeasonView;
  models: CompareModelId[];
  metric: string | undefined;
};

/** Explains that no season recap is selected, and how to build one. */
export function SeasonNotBuilt({ season }: { season: string }) {
  return (
    <section className="hm-card pf-empty" data-recap-section="season_to_date" data-season-recap-status="not_built">
      <div className="pf-eyebrow">Season to date</div>
      <h2>{unavailable("not_built", `No season recap is selected for ${season}`)}</h2>
      <p>
        Build it with <span className="mono">gridiron evaluate weekly-recap --season {season} --season-only</span>
      </p>
    </section>
  );
}

/** The season so far beside the closing market: scorecard, comparison chart, weekly recaps, calibration, wagers. */
export function SeasonScope({
  recap, state, onChange, onOpenWeek,
}: {
  recap: Season;
  state: SeasonState;
  onChange: (next: Partial<SeasonState>) => void;
  onOpenWeek: (week: number) => void;
}) {
  const scope = state.games === "all" ? recap.all_games : recap.live_games;
  const notBuilt = recap.weeks.filter((week) => week.status === "not_built");
  const lastWeek = Math.max(0, ...recap.weeks.filter((week) => week.status !== "not_built").map((week) => week.week));
  return (
    <div className="pf-scope season-panel" data-recap-section="season_to_date">
      <SeasonHeader recap={recap} scope={scope} state={state} onChange={onChange} lastWeek={lastWeek} notBuilt={notBuilt} />
      <section aria-labelledby="season-scorecard">
        <SectionHead
          eyebrow="Scorecard · season to date"
          title="Model vs. closing market"
          id="season-scorecard"
          description={`Every graded game pooled across ${scope.weeks} recapped week${scope.weeks === 1 ? "" : "s"}${state.games === "live" ? ", live forecasts only" : ""}.`}
        />
        <div className="pf-fam-grid">
          <FamilyScoreCard family="win" comparison={scope.metrics.win} />
          <FamilyScoreCard family="spread" comparison={scope.metrics.spread} />
          <FamilyScoreCard family="total" comparison={scope.metrics.total} />
          <FamilyScoreCard family="projected_score" comparison={scope.metrics.projected_score} />
        </div>
        <StartersLine scope={scope} />
      </section>
      <CompareChart
        eyebrow={`Compare · ${state.view === "week" ? "week by week" : "season to date"}`}
        title="Champion against other models"
        source={{ status: "available", data: seasonComparison(recap, state.games, state.view), metrics: SEASON_METRICS }}
        selected={state.models}
        onSelectedChange={(models) => onChange({ models })}
        metricKey={state.metric}
        onMetricChange={(metric) => onChange({ metric })}
        controls={
          <SegmentedControl
            label="Comparison period"
            options={[{ value: "week", label: "Each week" }, { value: "to_date", label: "To date" }]}
            value={state.view}
            onChange={(view) => onChange({ view })}
          />
        }
      />
      <WeeklyTable recap={recap} games={state.games} onOpenWeek={onOpenWeek} />
      <section className="hm-card pf-card" aria-labelledby="season-paired">
        <SectionHead
          eyebrow="Model minus market, per game"
          title="Is the difference more than noise?"
          id="season-paired"
          description={`Mean with an approximate 95% interval (mean ± ${recap.conventions.interval_z} × standard error, normal approximation)`}
        />
        <PairedPanel scope={scope} />
      </section>
      <section className="hm-card pf-card" aria-labelledby="season-reliability">
        <SectionHead eyebrow="Calibration" title="Reliability of the home win probability" id="season-reliability"
          description={recap.conventions.calibration_bins} />
        <ReliabilityPanel scope={scope} />
      </section>
      <SeasonWagers recap={recap} />
      <MethodDetails>
        <div>
          <span>Season metrics pool every graded game across the recapped weeks: {recap.conventions.season_to_date}.</span>
          <span>Live only: {recap.conventions.live_only}.</span>
          <span>Paired differences: {recap.conventions.paired_difference}; interval: {recap.conventions.interval}.</span>
        </div>
        <div>
          <span>Closing market: the closing spread as the expected home margin and the closing total; the market win
            probability is Φ(closing home margin / {recap.conventions.recap.margin_std}).</span>
        </div>
        <div className="pf-method-mono">
          <span>Season recap {recap.season_recap_id.slice(0, 12)}</span>
          <span>Built {formatTimestamp(recap.built_at)}</span>
          <span>Weekly recaps {recap.weekly_recap_ids.map((id) => id.slice(0, 8)).join(", ")}</span>
        </div>
      </MethodDetails>
    </div>
  );
}

function SeasonHeader({
  recap, scope, state, onChange, lastWeek, notBuilt,
}: {
  recap: Season;
  scope: Scope;
  state: SeasonState;
  onChange: (next: Partial<SeasonState>) => void;
  lastWeek: number;
  notBuilt: SeasonWeek[];
}) {
  const otherRoles = [...new Set(
    recap.weeks.filter((week) => week.status !== "not_built" && week.forecast_role !== "live").map((week) => week.forecast_role),
  )];
  const allLive = state.games === "live" || otherRoles.length === 0;
  const badge = state.games === "live"
    ? "Live forecasts only"
    : allLive
      ? "Live forecasts"
      : `Includes ${otherRoles.map((role) => role ?? "unrecorded").join(" and ")} weeks`;
  return (
    <header className="pf-head">
      <div className="pf-head-top">
        <div className="pf-eyebrow">Season recap</div>
        <SegmentedControl
          label="Forecasts included"
          options={[{ value: "all", label: "All weeks" }, { value: "live", label: "Live only" }]}
          value={state.games}
          onChange={(games) => onChange({ games })}
        />
      </div>
      <div className="pf-hero">
        <div>
          <div className="pf-badge-row">
            <span className={`pf-badge ${allLive ? "pf-badge-live" : "pf-badge-shadow"}`}>{badge}</span>
            <span className="pf-faint" style={{ fontSize: 11.5 }}>
              {scope.weeks} week{scope.weeks === 1 ? "" : "s"} · {scope.games} games
              {state.games === "live" && " · live forecasts only"}
            </span>
          </div>
          <h1>{recap.season} season <span>· {lastWeek > 1 ? `weeks 1–${lastWeek}` : `week ${lastWeek}`}</span></h1>
        </div>
        <CountsStrip
          label="Season counts"
          counts={[
            { label: "Weeks recapped", value: scope.weeks },
            { label: "Games graded", value: scope.games },
            { label: "Live-forecast games", value: recap.live_games.games, dim: true },
            { label: "Weeks not built", value: notBuilt.length, dim: true },
          ]}
        />
      </div>
      {notBuilt.map((week) => (
        <div key={week.week} data-season-week-status="not_built" style={{ fontSize: 12 }}>
          {unavailable("not_built", `Week ${week.week}: finished, but no weekly recap is selected`)}
        </div>
      ))}
    </header>
  );
}

function StartersLine({ scope }: { scope: Scope }) {
  const starters = scope.starting_qbs;
  return (
    <div className="pf-note" style={{ marginTop: 10 }} data-season-starters>
      Starting QBs: {starters.scored} of {starters.slots} slots scored
      {starters.scored > 0 && ` · assumed ${starters.assumed.correct}/${starters.assumed.count}`}
      {starters.not_recorded > 0 && ` · ${starters.not_recorded} not recorded`}
    </div>
  );
}

type TableFamily = { key: "win" | "spread" | "total"; label: string; digits: number; pick: (m: WeekMetrics) => [number | null | undefined, number | null | undefined, CloserCounts] };

const TABLE_FAMILIES: TableFamily[] = [
  { key: "win", label: "Moneyline Brier", digits: 3, pick: (m) => [m.win.model.brier, m.win.market.brier, m.win.closer] },
  { key: "spread", label: "Spread MAE", digits: 2, pick: (m) => [m.spread.model.mae, m.spread.market.mae, m.spread.closer] },
  { key: "total", label: "Total MAE", digits: 2, pick: (m) => [m.total.model.mae, m.total.market.mae, m.total.closer] },
];

/** One row per week; a row opens that week's recap. */
function WeeklyTable({ recap, games, onOpenWeek }: { recap: Season; games: SeasonGames; onOpenWeek: (week: number) => void }) {
  return (
    <section className="hm-card pf-tbl-card" aria-labelledby="season-weeks">
      <SectionHead eyebrow="Weekly recaps" title="Open any week for game detail" id="season-weeks" />
      <div className="pf-scroll" role="region" aria-label="Weekly recaps" tabIndex={0}>
        <table className="pf-table" style={{ minWidth: 980 }}>
          <caption className="visually-hidden">Each week's model and closing market metrics and how often the model was closer</caption>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>Week</th>
              <th scope="col" rowSpan={2}>Games</th>
              {TABLE_FAMILIES.map((family) => (
                <th key={family.key} scope="colgroup" colSpan={3} className="pf-grp">{family.label}</th>
              ))}
              <th scope="col" rowSpan={2}><span className="visually-hidden">Open</span></th>
            </tr>
            <tr className="pf-subhead">
              {TABLE_FAMILIES.map((family) => [
                <th key={`${family.key}-model`} scope="col" className="pf-grp">Model</th>,
                <th key={`${family.key}-market`} scope="col">Market</th>,
                <th key={`${family.key}-closer`} scope="col">Model closer</th>,
              ])}
            </tr>
          </thead>
          <tbody>
            {recap.weeks.map((week) => {
              const metrics = seasonWeekMetrics(week, games, "week");
              const status = week.status === "not_built" ? "Not built" : metrics ? null : "No live forecasts";
              const count = games === "all" ? week.games : week.live_games;
              return (
                <tr
                  key={week.week}
                  data-season-row={`week-${week.week}`}
                  data-clickable={status == null && week.status !== "not_built" ? "" : undefined}
                  onClick={week.status === "not_built" ? undefined : () => onOpenWeek(week.week)}
                >
                  <th scope="row">
                    {week.status === "not_built"
                      ? <span>Week {week.week}</span>
                      : <button type="button" className="pf-link" onClick={(event) => { event.stopPropagation(); onOpenWeek(week.week); }}>Week {week.week}</button>}
                    {week.forecast_role && week.forecast_role !== "live" && (
                      <span className="pf-sub pf-sub-warn">{week.forecast_role}</span>
                    )}
                  </th>
                  <td className="pf-num pf-muted">{week.status === "not_built" ? "—" : count}</td>
                  {status ? (
                    <td colSpan={9} className="pf-grp" style={{ textAlign: "left" }}>
                      {unavailable(week.status === "not_built" ? "not_built" : "no_live_forecasts", status)}
                    </td>
                  ) : (
                    TABLE_FAMILIES.map((family) => {
                      const [model, market, closer] = family.pick(metrics as WeekMetrics);
                      return <FamilyCells key={family.key} model={model} market={market} closer={closer} digits={family.digits} />;
                    })
                  )}
                  <td className="pf-open" aria-hidden="true">{week.status === "not_built" ? "" : "Recap →"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="pf-tbl-foot">
        <span>Lower is better for every error metric</span>
        <span>Green marks the better of model and market</span>
        <span>Model closer: games where the model was closer to the result, of all graded games</span>
      </div>
    </section>
  );
}

function FamilyCells({
  model, market, closer, digits,
}: {
  model: number | null | undefined;
  market: number | null | undefined;
  closer: CloserCounts;
  digits: number;
}) {
  const both = model != null && market != null;
  const modelBetter = both && model < market;
  const marketBetter = both && market < model;
  const total = closer.model + closer.market + closer.tie;
  return (
    <>
      <td className="pf-grp pf-num" style={{ color: modelBetter ? "var(--pos)" : marketBetter ? "var(--ink-3)" : undefined }}>
        {fixed(model, digits)}
      </td>
      <td className="pf-num" style={{ color: marketBetter ? "var(--pos)" : modelBetter ? "var(--ink-3)" : undefined }}>
        {fixed(market, digits)}
      </td>
      <td className="pf-num">
        {total === 0 ? "—" : (
          <>
            <span className="pf-mini" aria-hidden="true"><i style={{ width: `${(closer.model / total) * 100}%` }} /></span>
            {closer.model}/{total}
          </>
        )}
      </td>
    </>
  );
}

const PAIRED: [keyof Scope["paired"], string, number][] = [
  ["win", "Moneyline Brier", 4],
  ["spread", "Spread |error|", 2],
  ["total", "Total |error|", 2],
  ["projected_score", "Team score |error|", 2],
];

function PairedPanel({ scope }: { scope: Scope }) {
  return (
    <div className="season-whisker-grid pf-num" role="table" aria-label="Model minus market per game, with approximate 95% intervals">
      {PAIRED.map(([key, label, digits]) => (
        <WhiskerRow key={key} family={key} label={label} difference={scope.paired[key]} digits={digits} />
      ))}
      <div className="season-whisker-row pf-faint" style={{ fontSize: 10 }} aria-hidden="true">
        <span />
        <span className="season-axis">
          <span style={{ left: 0 }}>← model lower</span>
          <span style={{ left: "50%", transform: "translateX(-50%)" }}>0</span>
          <span style={{ left: "100%", transform: "translateX(-100%)" }}>market lower →</span>
        </span>
        <span />
      </div>
    </div>
  );
}

/** Each family on its own symmetric scale around zero (their units differ). */
function extentOf(difference: Paired): number {
  const extent = Math.max(
    ...[difference.mean, difference.interval_low, difference.interval_high].map((v) => Math.abs(v ?? 0)),
  );
  return extent > 0 ? extent * 1.15 : 1;
}

function WhiskerRow({
  family, label, difference, digits,
}: {
  family: string;
  label: string;
  difference: Paired;
  digits: number;
}) {
  const extent = extentOf(difference);
  const x = (value: number) => 50 + (value / extent) * 50;
  const { mean, interval_low: low, interval_high: high, count } = difference;
  let text: ReactNode;
  if (count === 0 || mean == null) text = unavailable("no_games", "No games both sides forecast");
  else if (low == null || high == null) text = <>{signed(mean, digits)} <span className="pf-faint">(n {count}; an interval needs two games)</span></>;
  else text = <>{signed(mean, digits)} <span className="pf-faint">[{signed(low, digits)}, {signed(high, digits)}] n {count}</span></>;
  return (
    <div role="row" tabIndex={0} className="season-whisker-row" data-season-paired={family}>
      <span role="rowheader" style={{ fontSize: 11.5, fontFamily: "var(--f-sans)" }}>{label}</span>
      <span role="cell" className="season-track">
        <span className="season-zero" />
        {low != null && high != null && (
          <span className="season-whisker" style={{ left: `${x(low)}%`, width: `${x(high) - x(low)}%` }} />
        )}
        {mean != null && <span className="season-dot season-dot-difference" style={{ left: `${x(mean)}%` }} />}
      </span>
      <span role="cell" style={{ fontSize: 11.5, textAlign: "right" }}>{text}</span>
    </div>
  );
}

function ReliabilityPanel({ scope }: { scope: Scope }) {
  const { model, market } = scope.calibration;
  const size = 240;
  const pad = 28;
  const plot = size - pad - 8;
  const px = (value: number) => pad + value * plot;
  const py = (value: number) => 8 + (1 - value) * plot;
  const maxCount = Math.max(1, ...model.bins.map((b) => b.count), ...market.bins.map((b) => b.count));
  const stripHeight = 40;
  return (
    <figure style={{ margin: 0 }} data-season-reliability>
      <figcaption className="pf-note">
        ECE model {fixed(model.ece ?? null, 3)} ({model.count}) · market {fixed(market.ece ?? null, 3)} ({market.count})
      </figcaption>
      <div style={{ display: "flex", gap: 8, fontSize: 11, marginTop: 8 }} className="pf-muted" aria-label="Legend">
        <span><span className="season-dot season-dot-model" style={{ position: "static", display: "inline-block", verticalAlign: "middle", transform: "none" }} /> Model</span>
        <span style={{ marginLeft: 12 }}><span className="season-dot season-dot-market" style={{ position: "static", display: "inline-block", verticalAlign: "middle", transform: "none" }} /> Closing market</span>
      </div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginTop: 10 }}>
        <div style={{ width: "100%", maxWidth: 320 }}>
          <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Observed home-win rate against mean forecast, by bin" style={{ display: "block", width: "100%" }}>
            {[0, 0.5, 1].map((tick) => (
              <g key={tick}>
                <line x1={px(0)} x2={px(1)} y1={py(tick)} y2={py(tick)} stroke="var(--line-soft)" strokeWidth={0.6} />
                <text x={pad - 4} y={py(tick) + 3} textAnchor="end" fontSize={8} fill="var(--ink-3)" fontFamily="var(--f-mono)">{tick}</text>
                <text x={px(tick)} y={size - 6} textAnchor="middle" fontSize={8} fill="var(--ink-3)" fontFamily="var(--f-mono)">{tick}</text>
              </g>
            ))}
            <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke="var(--line)" strokeWidth={1} />
            {sidePoints(market).map((b) => (
              <circle key={`market-${b.lower}`} cx={px(b.mean)} cy={py(b.observed)} r={4.5} fill="var(--bg-1)" stroke={MARKET} strokeWidth={2} data-reliability-point="market">
                <title>{binTitle("Market", b)}</title>
              </circle>
            ))}
            {sidePoints(model).map((b) => (
              <circle key={`model-${b.lower}`} cx={px(b.mean)} cy={py(b.observed)} r={4.5} fill={MODEL} stroke="var(--bg-1)" strokeWidth={2} data-reliability-point="model">
                <title>{binTitle("Model", b)}</title>
              </circle>
            ))}
          </svg>
          <svg viewBox={`0 0 ${size} ${stripHeight}`} role="img" aria-label="Decided games per bin, model and market" style={{ display: "block", width: "100%" }}>
            {model.bins.map((bin, index) => {
              const marketBin = market.bins[index];
              const center = px((bin.lower + bin.upper) / 2);
              const h = (count: number) => (count / maxCount) * (stripHeight - 12);
              return (
                <g key={bin.lower}>
                  <rect x={center - 7} y={stripHeight - 10 - h(bin.count)} width={6} height={h(bin.count)} rx={1} fill={MODEL}>
                    <title>{`Model ${range(bin)}: ${bin.count} games`}</title>
                  </rect>
                  <rect x={center + 1} y={stripHeight - 10 - h(marketBin.count)} width={6} height={h(marketBin.count)} rx={1} fill={MARKET}>
                    <title>{`Market ${range(marketBin)}: ${marketBin.count} games`}</title>
                  </rect>
                </g>
              );
            })}
            <line x1={px(0)} x2={px(1)} y1={stripHeight - 10} y2={stripHeight - 10} stroke="var(--line)" strokeWidth={0.6} />
            <text x={pad - 4} y={stripHeight - 10} textAnchor="end" fontSize={8} fill="var(--ink-3)">n</text>
          </svg>
          <div className="pf-faint" style={{ fontSize: 10.5, textAlign: "center" }}>Mean forecast in the bin</div>
        </div>
        <details style={{ flex: "1 1 280px" }}>
          <summary className="pf-muted" style={{ fontSize: 12, cursor: "pointer" }}>Bin table</summary>
          <div className="market-table-scroll" role="region" aria-label="Reliability bins" tabIndex={0}>
            <table className="recap-table" style={{ minWidth: 0 }}>
              <thead>
                <tr>
                  <th scope="col">Bin</th>
                  <th scope="col">Model n</th>
                  <th scope="col">Forecast</th>
                  <th scope="col">Home won</th>
                  <th scope="col">Market n</th>
                  <th scope="col">Forecast</th>
                  <th scope="col">Home won</th>
                </tr>
              </thead>
              <tbody>
                {model.bins.map((bin, index) => {
                  const marketBin = market.bins[index];
                  return (
                    <tr key={bin.lower} data-reliability-bin={index}>
                      <th scope="row">{range(bin)}</th>
                      <td>{bin.count}</td>
                      <td>{percent(bin.mean_forecast ?? null)}</td>
                      <td>{percent(bin.observed_rate ?? null)}</td>
                      <td>{marketBin.count}</td>
                      <td>{percent(marketBin.mean_forecast ?? null)}</td>
                      <td>{percent(marketBin.observed_rate ?? null)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </figure>
  );
}

/** Edges and recorded wagers week by week; the season's bets without a game come last. */
function SeasonWagers({ recap }: { recap: Season }) {
  const { recapped_weeks: total, unattributed } = recap.wagers;
  return (
    <section className="hm-card pf-card" data-season-wagers aria-labelledby="season-wagers">
      <SectionHead eyebrow="Edges and recorded wagers" title="By week" id="season-wagers" />
      <div className="market-table-scroll" role="region" aria-label="Edges and recorded wagers by week" tabIndex={0}>
        <table className="recap-table">
          <caption className="visually-hidden">Each week's edge evidence and recorded wagers, the season total, and bets without a game</caption>
          <thead>
            <tr>
              <th scope="col"><span className="visually-hidden">Week</span></th>
              <th scope="col">Edges</th>
              <th scope="col">Bets</th>
              <th scope="col">Won-lost-push</th>
              <th scope="col">Staked</th>
              <th scope="col">P&amp;L</th>
              <th scope="col">ROI</th>
            </tr>
          </thead>
          <tbody>
            {recap.weeks.map((week) => (
              <tr key={week.week} data-season-wager-week={week.week}>
                <th scope="row">Week {week.week}</th>
                {week.status === "not_built" || !week.wagers
                  ? <td colSpan={6}>{unavailable("not_built", "No weekly recap is selected")}</td>
                  : <>
                      <td title={week.edge_status ? recapEdgeStatusMessage(week.edge_status) : undefined}>
                        {week.edge_status ? recapEdgeStatusShort(week.edge_status) : "—"}
                      </td>
                      <WagerCells summary={week.wagers} />
                    </>}
              </tr>
            ))}
            <tr data-season-wager-week="total">
              <th scope="row">Recapped weeks</th>
              <td />
              <WagerCells summary={total} />
            </tr>
            <tr data-season-wager-week="unattributed">
              <th scope="row" title={recap.conventions.unattributed_wagers}>Without a game</th>
              <td />
              <WagerCells summary={unattributed} />
            </tr>
          </tbody>
        </table>
      </div>
      <div className="pf-note">
        Without a game: {recap.conventions.unattributed_wagers}
        {total.unsettled > 0 && ` · ${total.unsettled} recapped-week bet${total.unsettled === 1 ? "" : "s"} still open`}
      </div>
    </section>
  );
}

function WagerCells({ summary }: { summary: WagerSummary }) {
  return (
    <>
      <td>{summary.bets}</td>
      <td>{summary.bets === 0 ? "—" : record(summary)}</td>
      <td>{money(summary.staked)}</td>
      <td>{summary.bets === 0 ? "—" : signedMoney(summary.pnl)}</td>
      <td>{roi(summary.roi_pct)}</td>
    </>
  );
}

type Point = { lower: number; upper: number; count: number; mean: number; observed: number };

/** Bins with at least one game; an empty bin has no point to draw. */
function sidePoints(side: SideCalibration): Point[] {
  return side.bins.flatMap((bin) =>
    bin.count > 0 && bin.mean_forecast != null && bin.observed_rate != null
      ? [{ lower: bin.lower, upper: bin.upper, count: bin.count, mean: bin.mean_forecast, observed: bin.observed_rate }]
      : [],
  );
}

function binTitle(side: string, point: Point) {
  return `${side}, ${range(point)}: mean forecast ${percent(point.mean)}, home won ${percent(point.observed)} of ${point.count} games`;
}

function range(bin: { lower: number; upper: number }) {
  return `${bin.lower.toFixed(1)}–${bin.upper.toFixed(1)}`;
}

function unavailable(status: string, message: string) {
  return (
    <span data-weekly-status={status} data-weekly-value-state="unavailable" title={message} className="pf-status">
      {message}
    </span>
  );
}

function signed(value: number, digits: number) { return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`; }
