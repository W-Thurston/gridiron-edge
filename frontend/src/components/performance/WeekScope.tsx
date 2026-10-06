import type { ReactNode } from "react";
import type { components } from "../../api/schema";
import { WeeklyComponentValue } from "../field-status/WeeklyComponentValue";
import {
  recapActualSourceLabel,
  recapClosingLineMessage,
  recapFamilyLabel,
  recapForecastStatusMessage,
  recapForecastStatusShort,
  recapMarketAtForecastMessage,
  recapRecordSourceLabel,
  recapRoleLabel,
  recapStarterStatusMessage,
  recapStarterStatusShort,
} from "../field-status/recapStatus";
import type { RecapFamily } from "../field-status/recapStatus";
import { EdgesSection, WagersSection } from "../recap/MarketSections";
import { useNav } from "../../context/NavContext";
import { CountsStrip } from "./CountsStrip";
import { FamilyScoreCard } from "./FamilyScoreCard";
import { MethodDetails } from "./MethodDetails";
import { SectionHead } from "./SectionHead";
import { SegmentedControl } from "./SegmentedControl";
import { FAMILIES } from "./pageConstants";
import {
  formatKickoff, formatShortTimestamp, formatTimestamp, one, percent, signedOne, two,
} from "./format";

type Recap = components["schemas"]["WeeklyRecapResponse"];
type RecapSummary = components["schemas"]["WeeklyRecapSummary"];
type RecapGame = components["schemas"]["RecapGame"];
type Closer = components["schemas"]["Closer"];
type ForecastStatus = components["schemas"]["ForecastStatus"];
type ClosingLineStatus = components["schemas"]["ClosingLineStatus"];
type StarterSide = components["schemas"]["RecapStarterSide"];
type StarterStatus = components["schemas"]["StarterStatus"];
type PredictorAccuracy = components["schemas"]["RecapPredictorAccuracy"];


export type SelectedWeek = { season: string; week: number };

/** The week strip: every selected recap, labelled by season only when more than one season is listed. */
export function WeekStrip({
  items, selected, onSelect,
}: {
  items: RecapSummary[];
  selected: SelectedWeek;
  onSelect: (summary: RecapSummary) => void;
}) {
  const seasons = new Set(items.map((item) => item.season));
  return (
    <nav className="pf-weeks" aria-label="Recapped weeks">
      {items.map((item) => {
        const name = seasons.size > 1 ? `${item.season} · Week ${item.week}` : `Week ${item.week}`;
        return (
          <button
            key={`${item.season}-${item.week}`}
            type="button"
            aria-label={name}
            title={name}
            aria-pressed={item.season === selected.season && item.week === selected.week}
            onClick={() => onSelect(item)}
          >
            {seasons.size > 1 ? `${item.season.slice(2, 4)}·W${item.week}` : `W${item.week}`}
          </button>
        );
      })}
    </nav>
  );
}

export function RoleBadge({ role }: { role: Recap["forecast_role"] }) {
  return (
    <span className={`pf-badge ${role === "live" ? "pf-badge-live" : "pf-badge-shadow"}`} data-forecast-role={role ?? "unrecorded"}>
      {recapRoleLabel(role)}
    </span>
  );
}

/** The recap of one week: scorecard, every game, starting QBs, edges, wagers, then method and provenance. */
export function WeekScope({
  recap, strip, family, onFamily,
}: {
  recap: Recap;
  strip: ReactNode;
  family: RecapFamily;
  onFamily: (family: RecapFamily) => void;
}) {
  const { counts, conventions } = recap;
  return (
    <>
      <header className="pf-head">
        <div className="pf-head-top">
          <div className="pf-eyebrow">Weekly recap</div>
          {strip}
        </div>
        <div className="pf-hero">
          <div>
            <div className="pf-badge-row">
              <RoleBadge role={recap.forecast_role} />
              <span className="pf-faint" style={{ fontSize: 11.5 }}>Built {formatShortTimestamp(recap.built_at)}</span>
            </div>
            <h1>Week {recap.week} <span>· {recap.season}</span></h1>
          </div>
          <CountsStrip
            label="Game counts"
            counts={[
              { label: "Scheduled", value: counts.scheduled_games },
              { label: "In product", value: counts.in_product },
              { label: "Not in product", value: counts.not_in_product, dim: true },
              { label: "Carried", value: counts.carried_rows, dim: true },
              { label: "Tied", value: counts.outcome_ties, dim: true },
            ]}
          />
        </div>
      </header>
      <section aria-labelledby="week-scorecard">
        <SectionHead eyebrow="Scorecard" title="Model vs. closing market" id="week-scorecard" />
        <div className="pf-fam-grid">
          <FamilyScoreCard family="win" comparison={recap.metrics.win} />
          <FamilyScoreCard family="spread" comparison={recap.metrics.spread} />
          <FamilyScoreCard family="total" comparison={recap.metrics.total} />
          <FamilyScoreCard family="projected_score" comparison={recap.metrics.projected_score} />
        </div>
      </section>
      <section className="hm-card pf-tbl-card" aria-labelledby="week-games">
        <SectionHead eyebrow="Game by game" title="Every scheduled game" id="week-games">
          <SegmentedControl
            label="Forecast family"
            options={FAMILIES.map((value) => ({ value, label: recapFamilyLabel(value) }))}
            value={family}
            onChange={onFamily}
          />
        </SectionHead>
        <GameTable games={recap.games} family={family} />
        <div className="pf-tbl-foot">
          <span>Errors are forecast minus result</span>
          <span>Margins are home minus away</span>
          <span>Bars scale the absolute error; green marks the smaller of model and market</span>
        </div>
      </section>
      <StartersSection recap={recap} />
      <EdgesSection edges={recap.edges} />
      <WagersSection wagers={recap.wagers} />
      <MethodDetails>
        <div>
          Closing market: the closing spread as the expected home margin and the closing total; the market win
          probability is Φ(closing home margin / {conventions.margin_std}).
        </div>
        <div className="pf-method-mono">
          <span>Product {recap.product_id}</span>
          <span>Recap {recap.recap_id.slice(0, 12)}</span>
          <span>Built {formatTimestamp(recap.built_at)}</span>
        </div>
      </MethodDetails>
    </>
  );
}

type Column = {
  label: string;
  render: (game: RecapGame, scale: number) => ReactNode;
  /** Absolute errors in this column, for the shared bar scale. */
  error?: (game: RecapGame) => number | null | undefined;
};

function GameTable({ games, family }: { games: RecapGame[]; family: RecapFamily }) {
  const { navigate } = useNav();
  const columns = FAMILY_COLUMNS[family];
  const scale = Math.max(
    0,
    ...games.flatMap((game) => columns.map((column) => Math.abs(column.error?.(game) ?? 0))),
  );
  return (
    <div className="pf-scroll" role="region" aria-label="Recap games" tabIndex={0}>
      <table className="pf-table">
        <caption className="visually-hidden">
          {recapFamilyLabel(family)} forecast against the result and the closing market, one row per scheduled game
        </caption>
        <thead>
          <tr>
            <th scope="col">Matchup</th>
            <th scope="col">Final</th>
            {columns.map((column, index) => (
              <th key={column.label} scope="col" className={index === 0 || column.label === "Closer" ? "pf-grp" : undefined}
                style={column.label === "Closer" ? { textAlign: "center" } : undefined}>
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {games.map((game) => {
            const { away_score: away, home_score: home } = game.result;
            return (
              <tr key={game.game_id} data-product-row-status={game.product_row_status}>
                <th scope="row" style={{ fontWeight: 400 }}>
                  <button type="button" className="pf-match" aria-label={`${game.away_team} @ ${game.home_team}`}
                    onClick={() => navigate("/games", { gameId: game.game_id })}>
                    <span className="pf-team">{game.away_team}</span>
                    <span className="pf-at">@</span>
                    <span className="pf-team">{game.home_team}</span>
                  </button>
                  <span className="pf-sub">{formatKickoff(game.kickoff)}</span>
                  {game.forecast.carried && <span className="pf-sub">Carried from an earlier run</span>}
                  {game.forecast.role && game.forecast.role !== "live" && (
                    <span className="pf-sub pf-sub-warn" data-row-forecast-role={game.forecast.role}>
                      {game.forecast.role} forecast
                    </span>
                  )}
                </th>
                <td className="pf-final" aria-label={`${game.away_team} ${away}, ${game.home_team} ${home}`}>
                  <span className={home > away ? "pf-lose" : undefined}>{away}</span>
                  {" – "}
                  <span className={away > home ? "pf-lose" : undefined}>{home}</span>
                </td>
                {columns.map((column, index) => (
                  <td key={column.label} className={index === 0 || column.label === "Closer" ? "pf-grp" : undefined}
                    style={column.label === "Closer" ? { textAlign: "center" } : undefined}>
                    {column.render(game, scale)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const BOTH_CLOSING = (game: RecapGame): ClosingLineStatus =>
  game.closing_market.spread_status === "available" && game.closing_market.total_status === "available"
    ? "available"
    : "closing_line_missing";

const FAMILY_COLUMNS: Record<RecapFamily, Column[]> = {
  win: [
    { label: "Model P(home)", render: (game) => (
      <WeeklyComponentValue
        label="Moneyline"
        status={game.win.status}
        usable={game.win.status === "available" || game.win.status === "outcome_tie"}
        value={game.win.home_win_prob}
        format={(value) => `${(value * 100).toFixed(1)}%`}
        statusMessage={recapForecastStatusMessage("win", game.win.status)}
      />
    ) },
    { label: "Closing P(home)", render: (game) => (
      <span className="pf-muted">
        <WeeklyComponentValue
          label="Closing win probability"
          status={game.closing_market.spread_status}
          usable={game.closing_market.spread_status === "available"}
          value={game.closing_market.home_win_prob}
          format={(value) => `${(value * 100).toFixed(1)}%`}
          statusMessage={recapClosingLineMessage(game.closing_market.spread_status)}
        />
      </span>
    ) },
    { label: "Model Brier", error: (game) => game.win.model_brier, render: (game, scale) =>
      modelError(game.win.status, game.win.model_brier, closingOrNull(game.closing_market.spread_status, game.win.market_brier), scale, (v) => v.toFixed(3)) },
    { label: "Market Brier", error: (game) => game.win.market_brier, render: (game, scale) =>
      game.win.status === "outcome_tie"
        ? statusText("outcome_tie", recapForecastStatusMessage("win", "outcome_tie"), "Tied game")
        : closingError(game.closing_market.spread_status, game.win.market_brier, modelOrNull(game.win.status, game.win.model_brier), scale, (v) => v.toFixed(3)) },
    { label: "Closer", render: (game) => closer(game.win.closer, game.win.status, game.closing_market.spread_status, "win") },
  ],
  spread: [
    { label: "Model margin", render: (game) => modelValue("spread", game.spread.status, game.spread.value, signedOne) },
    { label: "Line at forecast", render: (game) => (
      <span className="pf-muted">
        <WeeklyComponentValue
          label="Spread at forecast"
          status={game.market_at_forecast.status}
          usable={game.market_at_forecast.status === "available"}
          value={game.market_at_forecast.home_margin}
          format={signedOne}
          statusMessage={recapMarketAtForecastMessage(game.market_at_forecast.status)}
        />
      </span>
    ) },
    { label: "Closing margin", render: (game) => closingValue(game.closing_market.spread_status, game.closing_market.home_margin, signedOne) },
    { label: "Model error", error: (game) => game.spread.model_error, render: (game, scale) =>
      modelError(game.spread.status, game.spread.model_error, closingOrNull(game.closing_market.spread_status, game.spread.market_error), scale, signedOne) },
    { label: "Market error", error: (game) => game.spread.market_error, render: (game, scale) =>
      closingError(game.closing_market.spread_status, game.spread.market_error, modelOrNull(game.spread.status, game.spread.model_error), scale, signedOne) },
    { label: "Closer", render: (game) => closer(game.spread.closer, game.spread.status, game.closing_market.spread_status, "spread") },
  ],
  total: [
    { label: "Model total", render: (game) => modelValue("total", game.total.status, game.total.value, one) },
    { label: "Line at forecast", render: (game) => (
      <span className="pf-muted">
        <WeeklyComponentValue
          label="Total at forecast"
          status={game.market_at_forecast.status}
          usable={game.market_at_forecast.status === "available"}
          value={game.market_at_forecast.total}
          format={one}
          statusMessage={recapMarketAtForecastMessage(game.market_at_forecast.status)}
        />
      </span>
    ) },
    { label: "Closing total", render: (game) => closingValue(game.closing_market.total_status, game.closing_market.total, one) },
    { label: "Model error", error: (game) => game.total.model_error, render: (game, scale) =>
      modelError(game.total.status, game.total.model_error, closingOrNull(game.closing_market.total_status, game.total.market_error), scale, signedOne) },
    { label: "Market error", error: (game) => game.total.market_error, render: (game, scale) =>
      closingError(game.closing_market.total_status, game.total.market_error, modelOrNull(game.total.status, game.total.model_error), scale, signedOne) },
    { label: "Closer", render: (game) => closer(game.total.closer, game.total.status, game.closing_market.total_status, "total") },
  ],
  projected_score: [
    { label: "Model score", render: (game) =>
      game.projected_score.status === "available" &&
      game.projected_score.away_score != null &&
      game.projected_score.home_score != null
        ? <span className="pf-num">{`${one(game.projected_score.away_score)} – ${one(game.projected_score.home_score)}`}</span>
        : statusText(
            game.projected_score.status,
            recapForecastStatusMessage("projected_score", game.projected_score.status),
          ) },
    { label: "Market implied", render: (game) => {
      const { implied_away_score: away, implied_home_score: home } = game.closing_market;
      return away != null && home != null
        ? <span className="pf-num pf-muted">{`${one(away)} – ${one(home)}`}</span>
        : statusText("closing_line_missing", recapClosingLineMessage("closing_line_missing"));
    } },
    { label: "Model error / team", error: (game) => game.projected_score.model_abs_error, render: (game, scale) =>
      modelError(game.projected_score.status, game.projected_score.model_abs_error,
        closingOrNull(BOTH_CLOSING(game), game.projected_score.market_abs_error), scale, two) },
    { label: "Market error / team", error: (game) => game.projected_score.market_abs_error, render: (game, scale) =>
      closingError(BOTH_CLOSING(game), game.projected_score.market_abs_error,
        modelOrNull(game.projected_score.status, game.projected_score.model_abs_error), scale, two) },
    { label: "Closer", render: (game) =>
      closer(game.projected_score.closer, game.projected_score.status, BOTH_CLOSING(game), "projected_score") },
  ],
};

function modelValue(
  family: RecapFamily,
  status: ForecastStatus,
  value: number | null | undefined,
  format: (value: number) => string,
) {
  return (
    <WeeklyComponentValue
      label={recapFamilyLabel(family)}
      status={status}
      usable={status === "available"}
      value={value}
      format={format}
      statusMessage={recapForecastStatusMessage(family, status)}
    />
  );
}

function closingValue(status: ClosingLineStatus, value: number | null | undefined, format: (value: number) => string) {
  return (
    <span className="pf-muted">
      <WeeklyComponentValue
        label="Closing line"
        status={status}
        usable={status === "available"}
        value={value}
        format={format}
        statusMessage={recapClosingLineMessage(status)}
      />
    </span>
  );
}

function modelOrNull(status: ForecastStatus, value: number | null | undefined) {
  return status === "available" ? value ?? null : null;
}
function closingOrNull(status: ClosingLineStatus, value: number | null | undefined) {
  return status === "available" ? value ?? null : null;
}

/** A model error cell: the value with its bar, or the family status in short form. */
function modelError(
  status: ForecastStatus,
  value: number | null | undefined,
  other: number | null,
  scale: number,
  format: (value: number) => string,
) {
  if (status === "available" && value != null) return <ErrorValue value={value} other={other} scale={scale} format={format} />;
  return statusText(status, recapForecastStatusShort(status), recapForecastStatusShort(status));
}

function closingError(
  status: ClosingLineStatus,
  value: number | null | undefined,
  other: number | null,
  scale: number,
  format: (value: number) => string,
) {
  if (status === "available" && value != null) return <ErrorValue value={value} other={other} scale={scale} format={format} />;
  return statusText(status, recapClosingLineMessage(status));
}

/** The error, a bar of its absolute size, and green when it is no larger than the other side's. */
function ErrorValue({
  value, other, scale, format,
}: {
  value: number;
  other: number | null;
  scale: number;
  format: (value: number) => string;
}) {
  const best = other == null || Math.abs(value) <= Math.abs(other);
  const width = scale > 0 ? Math.min(100, (Math.abs(value) / scale) * 100) : 0;
  return (
    <span className={`pf-err ${best && other != null ? "pf-err-best" : other != null ? "pf-err-worse" : ""}`}
      data-error-rank={other == null ? "alone" : best ? "smaller" : "larger"}>
      <span className="pf-err-bar" aria-hidden="true"><i style={{ width: `${width}%` }} /></span>
      <span>{format(value)}</span>
    </span>
  );
}

function closer(
  value: Closer | null | undefined,
  status: ForecastStatus,
  closingStatus: ClosingLineStatus,
  family: RecapFamily,
) {
  if (value) {
    return (
      <span className={`pf-pill pf-pill-${value}`} data-closer={value}>
        {value === "model" ? "Model" : value === "market" ? "Market" : "Tie"}
      </span>
    );
  }
  if (status !== "available") {
    return statusText(status, recapForecastStatusMessage(family, status), recapForecastStatusShort(status));
  }
  return statusText(closingStatus, recapClosingLineMessage(closingStatus));
}

/** Explicit text for a non-available value; the full message stays in the title. */
function statusText(status: string, message: string, text: string = message) {
  return (
    <span
      className="pf-status"
      data-weekly-status={status}
      data-weekly-value-state="unavailable"
      title={message}
      aria-label={message}
      style={{ fontSize: 11 }}
    >
      {text}
    </span>
  );
}

type Slot = { team: string; side: StarterSide };

/** Every starter slot, two per game, away first, in the recap's game order. */
function starterSlots(games: RecapGame[]): Slot[] {
  return games.flatMap((game) => [
    { team: game.away_team, side: game.starting_qbs.away },
    { team: game.home_team, side: game.starting_qbs.home },
  ]);
}

const PREDICTORS: [string, keyof Pick<Recap["starting_qbs"], "assumed" | "nflverse" | "carry_forward">][] = [
  ["Assumed by the forecast", "assumed"],
  ["nflverse at composition", "nflverse"],
  ["Carried forward", "carry_forward"],
];

function StartersSection({ recap }: { recap: Recap }) {
  const summary = recap.starting_qbs;
  const slots = starterSlots(recap.games);
  const misses = slots.filter(({ side }) =>
    side.status === "available" &&
    [side.assumed_correct, side.nflverse_correct, side.carry_forward_correct].includes(false),
  );
  const unscored = new Map<StarterStatus, string[]>();
  for (const { team, side } of slots) {
    if (side.status === "available") continue;
    unscored.set(side.status, [...(unscored.get(side.status) ?? []), team]);
  }
  const notes = [
    summary.from_capture > 0 && `${summary.from_capture} from the captured sidecar of their run`,
    summary.actual_from_game_logs > 0 &&
      `${summary.actual_from_game_logs} actual starter${summary.actual_from_game_logs === 1 ? "" : "s"} from the game logs`,
  ].filter(Boolean);
  return (
    <section className="hm-card pf-qb" data-recap-section="starting_qbs" aria-labelledby="week-starters">
      <div>
        <div>
          <div className="pf-eyebrow">Starting quarterbacks</div>
          <h2 id="week-starters">{summary.scored} of {summary.slots} team slots scored</h2>
          {notes.length > 0 && <p className="pf-muted" style={{ margin: "4px 0 0", fontSize: 11.5, lineHeight: 1.5 }}>{notes.join(" · ")}</p>}
        </div>
        <table className="pf-pred" aria-label="Starter predictor accuracy">
          <thead className="visually-hidden">
            <tr>
              <th scope="col">Predictor</th>
              <th scope="col">Accuracy bar</th>
              <th scope="col">Right</th>
              <th scope="col">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {PREDICTORS.map(([label, key]) => <PredictorRow key={key} label={label} accuracy={summary[key]} />)}
          </tbody>
        </table>
        <p className="pf-faint" style={{ margin: 0, fontSize: 11 }}>Who started: {recap.conventions.actual_starter}</p>
      </div>
      <div>
        <div className="pf-eyebrow">Misses · at least one predictor named the wrong starter</div>
        {misses.length > 0 && <MissTable misses={misses} />}
        {misses.length === 0 && summary.scored > 0 && (
          <div className="pf-muted" style={{ fontSize: 12 }}>Every predictor named every scored starter.</div>
        )}
        {misses.length === 0 && summary.scored === 0 && (
          <div className="pf-faint" style={{ fontSize: 12 }}>No starter slot was scored.</div>
        )}
        {unscored.size > 0 && (
          <div className="pf-unscored" aria-label="Unscored starter slots">
            {[...unscored].map(([status, teams]) => (
              <div key={status} className="pf-unscored-row" data-starter-status={status} title={recapStarterStatusMessage(status)}>
                <span>{recapStarterStatusShort(status)} ({teams.length})</span>
                <span className="pf-chips" aria-label={teams.join(", ")}>
                  {teams.map((team, index) => <span className="pf-chip" key={`${team}-${index}`} aria-hidden="true">{team}</span>)}
                </span>
                <span className="pf-faint" style={{ fontSize: 11 }}>{recapStarterStatusMessage(status)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function PredictorRow({ label, accuracy }: { label: string; accuracy: PredictorAccuracy }) {
  const share = accuracy.accuracy ?? null;
  return (
    <tr>
      <th scope="row">{label}</th>
      <td className="pf-pred-bar">
        <div className="bar-track" aria-hidden="true">
          <div className="bar-fill" style={{ width: `${(share ?? 0) * 100}%` }} />
        </div>
      </td>
      <td>{accuracy.correct} / {accuracy.count}</td>
      <td style={{ color: "var(--ink)" }}>{percent(share)}</td>
    </tr>
  );
}

function MissTable({ misses }: { misses: Slot[] }) {
  return (
    <div className="pf-scroll" role="region" aria-label="Starter misses" tabIndex={0}>
      <table className="pf-miss">
        <caption className="visually-hidden">Team slots where at least one predictor named the wrong starter</caption>
        <thead>
          <tr>
            <th scope="col">Team</th>
            <th scope="col">Assumed</th>
            <th scope="col">nflverse</th>
            <th scope="col">Carried forward</th>
            <th scope="col">Started</th>
          </tr>
        </thead>
        <tbody>
          {misses.map(({ team, side }) => (
            <tr key={`${team}-${side.actual_id}`}>
              <th scope="row"><span className="pf-team">{team}</span></th>
              <td>
                <Pick name={side.assumed_name} id={side.assumed_id} correct={side.assumed_correct} />
                <span className="pf-sub">
                  {side.assumed_source}
                  {side.assumed_as_of && ` · ${formatTimestamp(side.assumed_as_of)}`}
                  {side.record_source && ` · ${recapRecordSourceLabel(side.record_source)}`}
                </span>
              </td>
              <td><Pick name={side.nflverse_name} id={side.nflverse_id} correct={side.nflverse_correct} /></td>
              <td><Pick name={side.carried_forward_name} id={side.carried_forward_id} correct={side.carry_forward_correct} /></td>
              <td>
                <span style={{ fontWeight: 500 }}>{side.actual_name ?? side.actual_id}</span>
                {side.actual_source && <span className="pf-sub">{recapActualSourceLabel(side.actual_source)}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Pick({ name, id, correct }: { name?: string | null; id?: string | null; correct?: boolean | null }) {
  if (!id) return statusText("no_pick", "No pick recorded");
  const mark = correct === true ? "right" : correct === false ? "wrong" : "not scored";
  return (
    <span className="pf-pick" data-starter-pick={mark} title={`${name ?? id}: ${mark}`}>
      {name ?? id}
      <span className="visually-hidden"> ({mark})</span>
    </span>
  );
}
