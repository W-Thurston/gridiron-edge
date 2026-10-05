import { useState } from "react";
import type { ReactNode } from "react";
import { useSeasonRecap } from "../../api/hooks";
import type { components } from "../../api/schema";
import { ErrorCard } from "../error/ErrorCard";
import { RecapToggle } from "./RecapToggle";

type Season = components["schemas"]["SeasonRecapResponse"];
type SeasonWeek = components["schemas"]["SeasonRecapWeek"];
type Scope = components["schemas"]["SeasonRecapScope"];
type WeekMetrics = components["schemas"]["RecapWeekMetrics"];
type Paired = components["schemas"]["RecapPairedDifference"];
type SideCalibration = components["schemas"]["RecapSideCalibration"];

type ScopeKey = "all" | "live";
type View = "week" | "to_date";
type Pair = [number | null | undefined, number | null | undefined];

/** The model is the emphasized series, the closing market the neutral reference. */
const MODEL = "var(--series-model)";
const MARKET = "var(--series-market)";

const SERIES: { key: string; label: string; digits: number; pick: (metrics: WeekMetrics) => Pair }[] = [
  { key: "win", label: "Win Brier", digits: 3, pick: (m) => [m.win.model.brier, m.win.market.brier] },
  { key: "spread", label: "Spread MAE", digits: 2, pick: (m) => [m.spread.model.mae, m.spread.market.mae] },
  { key: "total", label: "Total MAE", digits: 2, pick: (m) => [m.total.model.mae, m.total.market.mae] },
];

/** The season so far, beside the market: week by week, season to date, calibration. */
export function SeasonPanel({ season }: { season: string }) {
  const query = useSeasonRecap(season);
  if (query.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading the season recap…</div>;
  }
  if (query.error) return <ErrorCard error={query.error} onRetry={() => query.refetch()} />;
  if (!query.data) {
    return (
      <section className="hm-card" style={{ padding: 20 }} data-recap-section="season_to_date">
        <div className="upper dim" style={{ fontSize: 10 }}>Season to date</div>
        <div style={{ marginTop: 8 }} data-season-recap-status="not_built">
          {unavailable("not_built", `No season recap is selected for ${season}`)}
          <div className="dim2" style={{ fontSize: 11, marginTop: 6 }}>
            Build it with <span className="mono">gridiron evaluate weekly-recap --season {season} --season-only</span>
          </div>
        </div>
      </section>
    );
  }
  return <SeasonBody recap={query.data} />;
}

function SeasonBody({ recap }: { recap: Season }) {
  const [scopeKey, setScopeKey] = useState<ScopeKey>("all");
  const [view, setView] = useState<View>("week");
  const scope = scopeKey === "all" ? recap.all_games : recap.live_games;
  const notBuilt = recap.weeks.filter((week) => week.status === "not_built");
  return (
    <section className="hm-card season-panel" style={{ padding: 20 }} data-recap-section="season_to_date">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div className="upper dim" style={{ fontSize: 10 }}>Season to date · {recap.season}</div>
          <div className="mono dim" style={{ fontSize: 11, marginTop: 6 }}>
            {scope.weeks} week{scope.weeks === 1 ? "" : "s"} · {scope.games} games
            {scopeKey === "live" && " · live forecasts only"}
          </div>
          {notBuilt.map((week) => (
            <div key={week.week} data-season-week-status="not_built" style={{ fontSize: 11, marginTop: 4 }}>
              {unavailable("not_built", `Week ${week.week}: finished, but no weekly recap is selected`)}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "flex-start" }}>
          <RecapToggle active={view === "week"} onClick={() => setView("week")}>Week by week</RecapToggle>
          <RecapToggle active={view === "to_date"} onClick={() => setView("to_date")}>Season to date</RecapToggle>
          <span style={{ width: 8 }} />
          <RecapToggle active={scopeKey === "all"} onClick={() => setScopeKey("all")}>All weeks</RecapToggle>
          <RecapToggle active={scopeKey === "live"} onClick={() => setScopeKey("live")}>Live only</RecapToggle>
        </div>
      </div>
      <Legend />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginTop: 12 }}>
        {SERIES.map((series) => (
          <DumbbellPanel key={series.key} series={series} recap={recap} scope={scope} scopeKey={scopeKey} view={view} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20, marginTop: 20 }}>
        <SeasonTotals scope={scope} />
        <PairedPanel scope={scope} recap={recap} />
      </div>
      <ReliabilityPanel scope={scope} recap={recap} />
    </section>
  );
}

function Legend() {
  return (
    <div className="mono dim" style={{ display: "flex", gap: 16, fontSize: 10, marginTop: 12 }} aria-label="Legend">
      <span><Dot side="model" /> Model</span>
      <span><Dot side="market" /> Closing market</span>
    </div>
  );
}

function Dot({ side }: { side: "model" | "market" }) {
  return <span className={`season-dot season-dot-${side}`} style={{ position: "static", display: "inline-block", verticalAlign: "middle" }} />;
}

type Row = { key: string; label: string; role?: string | null; games?: number; metrics: WeekMetrics | null; status?: string };

/** The week's metrics in the chosen scope and view, or the reason there are none. */
function weekRow(week: SeasonWeek, scopeKey: ScopeKey, view: View): Row {
  const base = { key: `week-${week.week}`, label: `Week ${week.week}`, role: week.forecast_role };
  if (week.status === "not_built") return { ...base, metrics: null, status: "Not built" };
  if (view === "week" && scopeKey === "live" && week.live_games === 0) {
    return { ...base, metrics: null, status: "No live forecasts" };
  }
  const metrics = view === "week"
    ? (scopeKey === "all" ? week.metrics : week.live_metrics)
    : (scopeKey === "all" ? week.to_date : week.live_to_date);
  const games = view === "week" ? (scopeKey === "all" ? week.games : week.live_games) : undefined;
  return { ...base, metrics: metrics ?? null, games };
}

function DumbbellPanel({
  series, recap, scope, scopeKey, view,
}: {
  series: (typeof SERIES)[number];
  recap: Season;
  scope: Scope;
  scopeKey: ScopeKey;
  view: View;
}) {
  const rows: Row[] = [
    ...recap.weeks.map((week) => weekRow(week, scopeKey, view)),
    { key: "season", label: "Season", metrics: scope.metrics },
  ];
  const values = rows.flatMap((row) => (row.metrics ? series.pick(row.metrics) : []))
    .filter((value): value is number => value != null);
  const [lo, hi] = domain(values);
  const x = (value: number) => ((value - lo) / (hi - lo)) * 100;
  const format = (value: number | null | undefined) => (value == null ? "—" : value.toFixed(series.digits));
  return (
    <figure style={{ margin: 0 }} data-season-series={series.key}>
      <figcaption className="upper dim" style={{ fontSize: 9 }}>
        {series.label} · {view === "week" ? "each week" : "season to date"} · lower is better
      </figcaption>
      <div className="season-dumbbell-grid mono tnum" role="table" aria-label={`${series.label}, model and closing market`}>
        <div role="row" className="season-dumbbell-row dim2" style={{ fontSize: 9 }}>
          <span role="columnheader" />
          <span role="columnheader" className="visually-hidden">Range</span>
          <span role="columnheader" style={{ textAlign: "right" }}>Model</span>
          <span role="columnheader" style={{ textAlign: "right" }}>Market</span>
        </div>
        {rows.map((row) => {
          const [model, market] = row.metrics ? series.pick(row.metrics) : [null, null];
          const status = row.status ?? (model == null && market == null ? "No games scored" : undefined);
          return (
            <div
              key={row.key}
              role="row"
              tabIndex={0}
              className={`season-dumbbell-row${row.key === "season" ? " season-dumbbell-total" : ""}`}
              data-season-row={row.key}
              title={status ? `${row.label}: ${status}` : `${row.label}: model ${format(model)}, market ${format(market)}`}
            >
              <span role="rowheader" style={{ fontSize: 11 }}>
                {row.label}
                {row.role && row.role !== "live" && <span style={{ color: "var(--warn)", fontSize: 9 }}> {row.role}</span>}
              </span>
              <span role="cell" className="season-track">
                {status ? (
                  unavailable(row.status ? "not_built" : "no_games", status)
                ) : (
                  <>
                    {model != null && market != null && (
                      <span className="season-link" style={{ left: `${Math.min(x(model), x(market))}%`, width: `${Math.abs(x(model) - x(market))}%` }} />
                    )}
                    {market != null && <span className="season-dot season-dot-market" style={{ left: `${x(market)}%` }} />}
                    {model != null && <span className="season-dot season-dot-model" style={{ left: `${x(model)}%` }} />}
                  </>
                )}
              </span>
              <span role="cell" style={{ textAlign: "right", fontSize: 11 }}>{status ? "" : format(model)}</span>
              <span role="cell" style={{ textAlign: "right", fontSize: 11, color: "var(--ink-3)" }}>{status ? "" : format(market)}</span>
            </div>
          );
        })}
        <div className="season-dumbbell-row dim2" style={{ fontSize: 9 }} aria-hidden="true">
          <span />
          <span className="season-axis">
            <span style={{ left: 0 }}>{lo.toFixed(series.digits)}</span>
            <span style={{ left: "100%", transform: "translateX(-100%)" }}>{hi.toFixed(series.digits)}</span>
          </span>
          <span />
          <span />
        </div>
      </div>
    </figure>
  );
}

function SeasonTotals({ scope }: { scope: Scope }) {
  const { win, spread, total, projected_score: score } = scope.metrics;
  const rows: [string, string, string][] = [
    ["Win Brier", withCount(win.model.brier, 4, win.model.count), withCount(win.market.brier, 4, win.market.count)],
    ["Log loss", fixed(win.model.log_loss, 4), fixed(win.market.log_loss, 4)],
    ["Accuracy", percent(win.model.accuracy), percent(win.market.accuracy)],
    ["Spread MAE", withCount(spread.model.mae, 2, spread.model.count), withCount(spread.market.mae, 2, spread.market.count)],
    ["Total MAE", withCount(total.model.mae, 2, total.model.count), withCount(total.market.mae, 2, total.market.count)],
    ["Team score MAE", withCount(score.model.mae, 2, score.model.count), withCount(score.market.mae, 2, score.market.count)],
  ];
  const starters = scope.starting_qbs;
  return (
    <div>
      <div className="upper dim" style={{ fontSize: 9 }}>Season totals</div>
      <table className="mono tnum" style={{ width: "100%", marginTop: 8, fontSize: 12, borderCollapse: "collapse" }} aria-label="Season totals, model and closing market">
        <thead>
          <tr className="dim" style={{ fontSize: 9 }}>
            <th scope="col" style={{ textAlign: "left", fontWeight: 400 }} />
            <th scope="col" style={{ textAlign: "right", fontWeight: 400 }}>Model (n)</th>
            <th scope="col" style={{ textAlign: "right", fontWeight: 400 }}>Market (n)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, model, market]) => (
            <tr key={label}>
              <th scope="row" className="dim" style={{ textAlign: "left", fontWeight: 400, padding: "3px 0" }}>{label}</th>
              <td style={{ textAlign: "right" }}>{model}</td>
              <td style={{ textAlign: "right" }}>{market}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mono dim" style={{ fontSize: 10, marginTop: 8 }}>
        Closer to the result: win {win.closer.model}/{win.closer.market}/{win.closer.tie} · spread{" "}
        {spread.closer.model}/{spread.closer.market}/{spread.closer.tie} · total {total.closer.model}/
        {total.closer.market}/{total.closer.tie} (model/market/tie)
      </div>
      <div className="mono dim" style={{ fontSize: 10, marginTop: 4 }} data-season-starters>
        Starting QBs: {starters.scored} of {starters.slots} slots scored
        {starters.scored > 0 && ` · assumed ${starters.assumed.correct}/${starters.assumed.count}`}
        {starters.not_recorded > 0 && ` · ${starters.not_recorded} not recorded`}
      </div>
    </div>
  );
}

const PAIRED: [keyof Scope["paired"], string, number][] = [
  ["win", "Win Brier", 4],
  ["spread", "Spread |error|", 2],
  ["total", "Total |error|", 2],
  ["projected_score", "Team score |error|", 2],
];

function PairedPanel({ scope, recap }: { scope: Scope; recap: Season }) {
  return (
    <div>
      <div className="upper dim" style={{ fontSize: 9 }}>Model minus market, per game</div>
      <div className="mono dim2" style={{ fontSize: 9, marginTop: 4 }}>
        Mean with an approximate 95% interval (mean ± {recap.conventions.interval_z} × standard error, normal approximation)
      </div>
      <div className="season-whisker-grid mono tnum" role="table" aria-label="Model minus market per game, with approximate 95% intervals" style={{ marginTop: 10 }}>
        {PAIRED.map(([key, label, digits]) => (
          <WhiskerRow key={key} family={key} label={label} difference={scope.paired[key]} digits={digits} />
        ))}
        <div className="season-whisker-row dim2" style={{ fontSize: 9 }} aria-hidden="true">
          <span />
          <span className="season-axis">
            <span style={{ left: 0 }}>← model lower</span>
            <span style={{ left: "50%", transform: "translateX(-50%)" }}>0</span>
            <span style={{ left: "100%", transform: "translateX(-100%)" }}>market lower →</span>
          </span>
          <span />
        </div>
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
  else if (low == null || high == null) text = <>{signed(mean, digits)} <span className="dim2">(n {count}; an interval needs two games)</span></>;
  else text = <>{signed(mean, digits)} <span className="dim2">[{signed(low, digits)}, {signed(high, digits)}] n {count}</span></>;
  return (
    <div role="row" tabIndex={0} className="season-whisker-row" data-season-paired={family}>
      <span role="rowheader" style={{ fontSize: 11 }}>{label}</span>
      <span role="cell" className="season-track">
        <span className="season-zero" />
        {low != null && high != null && (
          <span className="season-whisker" style={{ left: `${x(low)}%`, width: `${x(high) - x(low)}%` }} />
        )}
        {mean != null && <span className="season-dot season-dot-difference" style={{ left: `${x(mean)}%` }} />}
      </span>
      <span role="cell" style={{ fontSize: 11, textAlign: "right" }}>{text}</span>
    </div>
  );
}

function ReliabilityPanel({ scope, recap }: { scope: Scope; recap: Season }) {
  const { model, market } = scope.calibration;
  const size = 240;
  const pad = 28;
  const plot = size - pad - 8;
  const px = (value: number) => pad + value * plot;
  const py = (value: number) => 8 + (1 - value) * plot;
  const maxCount = Math.max(1, ...model.bins.map((b) => b.count), ...market.bins.map((b) => b.count));
  const stripHeight = 40;
  return (
    <figure style={{ margin: "20px 0 0" }} data-season-reliability>
      <figcaption className="upper dim" style={{ fontSize: 9 }}>
        Reliability · home win probability · ECE model {fixed(model.ece ?? null, 3)} ({model.count}) · market{" "}
        {fixed(market.ece ?? null, 3)} ({market.count})
      </figcaption>
      <div className="mono dim2" style={{ fontSize: 9, marginTop: 4 }}>{recap.conventions.calibration_bins}</div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginTop: 10 }}>
        <div style={{ width: "100%", maxWidth: 320 }}>
          <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Observed home-win rate against mean forecast, by bin" style={{ display: "block", width: "100%" }}>
            {[0, 0.5, 1].map((tick) => (
              <g key={tick}>
                <line x1={px(0)} x2={px(1)} y1={py(tick)} y2={py(tick)} stroke="var(--line-soft)" strokeWidth={0.6} />
                <text x={pad - 4} y={py(tick) + 3} textAnchor="end" fontSize={8} fill="var(--ink-3)">{tick}</text>
                <text x={px(tick)} y={size - 6} textAnchor="middle" fontSize={8} fill="var(--ink-3)">{tick}</text>
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
          <div className="mono dim2" style={{ fontSize: 9, textAlign: "center" }}>Mean forecast in the bin</div>
        </div>
        <details style={{ flex: "1 1 280px" }}>
          <summary className="mono dim" style={{ fontSize: 10, cursor: "pointer" }}>Bin table</summary>
          <div className="market-table-scroll" role="region" aria-label="Reliability bins" tabIndex={0}>
            <table className="recap-table mono tnum" style={{ minWidth: 0 }}>
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

/** A padded [low, high] domain over the drawn values. */
function domain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.1 || Math.abs(max) * 0.05 || 0.01;
  return [min - pad, max + pad];
}

function unavailable(status: string, message: string) {
  return (
    <span data-weekly-status={status} data-weekly-value-state="unavailable" title={message} style={{ color: "var(--ink-4)" }}>
      {message}
    </span>
  );
}

function withCount(value: number | null | undefined, digits: number, count: number) {
  return `${fixed(value ?? null, digits)} (${count})`;
}
function fixed(value: number | null, digits: number) { return value == null ? "—" : value.toFixed(digits); }
function percent(value: number | null | undefined) { return value == null ? "—" : `${(value * 100).toFixed(1)}%`; }
function signed(value: number, digits: number) { return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`; }
