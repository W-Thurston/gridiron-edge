import { useState } from "react";
import type { ReactNode } from "react";
import { useRecaps, useWeeklyRecap } from "../api/hooks";
import type { components } from "../api/schema";
import { ErrorCard } from "../components/error/ErrorCard";
import { FieldValue } from "../components/field-status/FieldValue";
import { WeeklyComponentValue } from "../components/field-status/WeeklyComponentValue";
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
} from "../components/field-status/recapStatus";
import type { RecapFamily } from "../components/field-status/recapStatus";
import { useNav } from "../context/NavContext";

type Recap = components["schemas"]["WeeklyRecapResponse"];
type RecapSummary = components["schemas"]["WeeklyRecapSummary"];
type RecapGame = components["schemas"]["RecapGame"];
type Closer = components["schemas"]["Closer"];
type ForecastStatus = components["schemas"]["ForecastStatus"];
type ProbabilityMetrics = components["schemas"]["RecapProbabilityMetrics"];
type ErrorMetrics = components["schemas"]["RecapErrorMetrics"];
type CloserCounts = components["schemas"]["RecapCloserCounts"];
type StarterSide = components["schemas"]["RecapStarterSide"];
type StarterStatus = components["schemas"]["StarterStatus"];
type PredictorAccuracy = components["schemas"]["RecapPredictorAccuracy"];

const FAMILIES: RecapFamily[] = ["win", "spread", "total", "projected_score"];

const PENDING_SECTIONS = [
  ["edges", "Edges and recommendations"],
  ["wagers", "Recorded wagers"],
  ["season_to_date", "Season to date"],
] as const;

export function WeeklyRecap() {
  const { route, navigate } = useNav();
  const recaps = useRecaps();
  const items = recaps.data?.items ?? [];
  const selected = selectedWeek(route.params, items);
  const recap = useWeeklyRecap(selected?.season ?? null, selected?.week ?? null);

  if (recaps.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading weekly recaps…</div>;
  }
  if (recaps.error) {
    return <ErrorCard error={recaps.error} onRetry={() => recaps.refetch()} />;
  }
  if (!selected) return <NoRecaps />;

  const choose = (summary: RecapSummary) =>
    navigate("/recap", { season: summary.season, week: String(summary.week) });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <WeekPicker items={items} selected={selected} onSelect={choose} />
      {recap.isLoading && (
        <div className="hm-card" style={{ padding: 24 }}>Loading the week {selected.week} recap…</div>
      )}
      {recap.error && <ErrorCard error={recap.error} onRetry={() => recap.refetch()} />}
      {recap.data && <RecapBody recap={recap.data} />}
    </div>
  );
}

/** The week named by the route, else the highest (season, week) among the selected recaps. */
function selectedWeek(
  params: Record<string, string>,
  items: RecapSummary[],
): { season: string; week: number } | null {
  const week = Number(params.week);
  if (params.season && Number.isInteger(week) && week >= 1) {
    return { season: params.season, week };
  }
  const latest = [...items].sort(
    (a, b) => a.season.localeCompare(b.season) || a.week - b.week,
  ).at(-1);
  return latest ? { season: latest.season, week: latest.week } : null;
}

function NoRecaps() {
  return (
    <section className="hm-card" style={{ padding: 32 }}>
      <div className="upper dim" style={{ fontSize: 10 }}>Last Week</div>
      <h2 style={{ margin: "10px 0 8px" }}>No weekly recap has been built yet</h2>
      <p className="dim" style={{ maxWidth: 720, lineHeight: 1.6 }}>
        A recap is built once every game of a week has a result:{" "}
        <span className="mono">gridiron evaluate weekly-recap --season S --week W</span>
      </p>
    </section>
  );
}

function WeekPicker({
  items,
  selected,
  onSelect,
}: {
  items: RecapSummary[];
  selected: { season: string; week: number };
  onSelect: (summary: RecapSummary) => void;
}) {
  const seasons = new Set(items.map((item) => item.season));
  return (
    <section className="hm-card" style={{ padding: 16 }} aria-label="Recapped weeks">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {items.map((item) => (
          <Toggle
            key={`${item.season}-${item.week}`}
            active={item.season === selected.season && item.week === selected.week}
            onClick={() => onSelect(item)}
          >
            {seasons.size > 1 ? `${item.season} · Week ${item.week}` : `Week ${item.week}`}
          </Toggle>
        ))}
      </div>
    </section>
  );
}

function RecapBody({ recap }: { recap: Recap }) {
  const [family, setFamily] = useState<RecapFamily>("win");
  const fieldStatus = recap._meta?.field_status ?? {};
  return (
    <>
      <RecapHeader recap={recap} />
      <section
        style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}
      >
        <WinCard metrics={recap.metrics.win} />
        <PointCard family="spread" comparison={recap.metrics.spread} />
        <PointCard family="total" comparison={recap.metrics.total} />
        <PointCard family="projected_score" comparison={recap.metrics.projected_score} />
      </section>
      <section className="hm-card" style={{ padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div className="upper dim" style={{ fontSize: 10 }}>Every scheduled game</div>
            <div className="mono dim2" style={{ fontSize: 9, marginTop: 4 }}>
              Errors are forecast minus result · margins are home minus away
            </div>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {FAMILIES.map((value) => (
              <Toggle key={value} active={family === value} onClick={() => setFamily(value)}>
                {recapFamilyLabel(value)}
              </Toggle>
            ))}
          </div>
        </div>
        <GameTable games={recap.games} family={family} />
      </section>
      <StartersSection recap={recap} />
      <section
        style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}
      >
        {PENDING_SECTIONS.map(([key, label]) => (
          <div key={key} className="hm-card" style={{ padding: 18 }} data-recap-section={key}>
            <div className="upper dim" style={{ fontSize: 9 }}>{label}</div>
            <div style={{ marginTop: 8 }}>
              <FieldValue value={null} status={fieldStatus[key]} placeholder="Not in the recap yet" />
            </div>
          </div>
        ))}
      </section>
    </>
  );
}

function RecapHeader({ recap }: { recap: Recap }) {
  const { counts, conventions } = recap;
  return (
    <section className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10 }}>Last Week</div>
      <h1 style={{ margin: "8px 0 6px", fontSize: 28 }}>
        {recap.season} week {recap.week}
      </h1>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <RoleBadge role={recap.forecast_role} />
        <span className="mono dim" style={{ fontSize: 11 }}>
          {counts.scheduled_games} scheduled · {counts.in_product} in product ·{" "}
          {counts.not_in_product} not in product · {counts.carried_rows} carried ·{" "}
          {counts.outcome_ties} tied
        </span>
      </div>
      <div className="dim" style={{ fontSize: 12, marginTop: 10, lineHeight: 1.5 }}>
        Closing market: the closing spread as the expected home margin and the closing total; the
        market win probability is Φ(closing home margin / {conventions.margin_std}).
      </div>
      <div className="mono dim2" style={{ fontSize: 9, marginTop: 6 }}>
        Product {recap.product_id} · recap {recap.recap_id.slice(0, 12)} · built{" "}
        {formatTimestamp(recap.built_at)}
      </div>
    </section>
  );
}

function RoleBadge({ role }: { role: Recap["forecast_role"] }) {
  const color = role === "live" ? "var(--pos)" : "var(--warn)";
  return (
    <span
      className="mono"
      data-forecast-role={role ?? "unrecorded"}
      style={{
        fontSize: 10,
        padding: "2px 8px",
        borderRadius: 3,
        border: `1px solid ${color}`,
        color,
      }}
    >
      {recapRoleLabel(role)}
    </span>
  );
}

function WinCard({ metrics }: { metrics: Recap["metrics"]["win"] }) {
  const rows: [string, (side: ProbabilityMetrics) => string][] = [
    ["Brier", (side) => fixed(side.brier, 4)],
    ["Log loss", (side) => fixed(side.log_loss, 4)],
    ["Accuracy", (side) => percent(side.accuracy)],
    ["Games", (side) => String(side.count)],
  ];
  return <MetricCard family="win" rows={rows} model={metrics.model} market={metrics.market} closer={metrics.closer} />;
}

function PointCard({
  family,
  comparison,
}: {
  family: Exclude<RecapFamily, "win">;
  comparison: Recap["metrics"]["spread"];
}) {
  const rows: [string, (side: ErrorMetrics) => string][] = [
    ["MAE", (side) => fixed(side.mae, 2)],
    ["RMSE", (side) => fixed(side.rmse, 2)],
    ["Bias", (side) => signedFixed(side.bias, 2)],
    [family === "projected_score" ? "Team scores" : "Games", (side) => String(side.count)],
  ];
  return (
    <MetricCard
      family={family}
      rows={rows}
      model={comparison.model}
      market={comparison.market}
      closer={comparison.closer}
    />
  );
}

function MetricCard<T>({
  family,
  rows,
  model,
  market,
  closer,
}: {
  family: RecapFamily;
  rows: [string, (side: T) => string][];
  model: T;
  market: T;
  closer: CloserCounts;
}) {
  return (
    <div className="hm-card" style={{ padding: 18 }} data-recap-family={family}>
      <div className="upper dim" style={{ fontSize: 10 }}>{recapFamilyLabel(family)}</div>
      <table className="mono tnum" style={{ width: "100%", marginTop: 10, fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr className="dim" style={{ fontSize: 9 }}>
            <th scope="col" style={{ textAlign: "left", fontWeight: 400 }} />
            <th scope="col" style={{ textAlign: "right", fontWeight: 400 }}>Model</th>
            <th scope="col" style={{ textAlign: "right", fontWeight: 400 }}>Market</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th scope="row" className="dim" style={{ textAlign: "left", fontWeight: 400, padding: "3px 0" }}>{label}</th>
              <td style={{ textAlign: "right" }}>{value(model)}</td>
              <td style={{ textAlign: "right" }}>{value(market)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mono dim" style={{ fontSize: 10, marginTop: 10 }}>
        Closer to the result: model {closer.model} · market {closer.market} · tie {closer.tie}
      </div>
    </div>
  );
}

function GameTable({ games, family }: { games: RecapGame[]; family: RecapFamily }) {
  const { navigate } = useNav();
  const columns = FAMILY_COLUMNS[family];
  return (
    <div className="market-table-scroll" role="region" aria-label="Recap games" tabIndex={0} style={{ marginTop: 14 }}>
      <table className="recap-table mono tnum">
        <caption className="visually-hidden">
          {recapFamilyLabel(family)} forecast against the result and the closing market, one row per scheduled game
        </caption>
        <thead>
          <tr>
            <th scope="col">Matchup</th>
            <th scope="col">Kickoff</th>
            <th scope="col">Final</th>
            {columns.map(([label]) => (
              <th key={label} scope="col">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {games.map((game) => (
            <tr key={game.game_id} data-product-row-status={game.product_row_status}>
              <th scope="row">
                <button
                  type="button"
                  className="recap-game-link"
                  onClick={() => navigate("/games", { gameId: game.game_id })}
                >
                  {game.away_team} @ {game.home_team}
                </button>
                {game.forecast.carried && <div className="dim2" style={{ fontSize: 9 }}>carried from an earlier run</div>}
                {game.forecast.role && game.forecast.role !== "live" && (
                  <div style={{ fontSize: 9, color: "var(--warn)" }}>{game.forecast.role}</div>
                )}
              </th>
              <td>{formatKickoff(game.kickoff)}</td>
              <td>
                {game.away_team} {game.result.away_score} – {game.home_team} {game.result.home_score}
              </td>
              {columns.map(([label, render]) => (
                <td key={label}>{render(game)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
  return (
    <section className="hm-card" style={{ padding: 20 }} data-recap-section="starting_qbs">
      <div className="upper dim" style={{ fontSize: 10 }}>Starting quarterbacks</div>
      <div className="mono dim" style={{ fontSize: 11, marginTop: 6 }}>
        {summary.scored} of {summary.slots} team slots scored
        {summary.from_capture > 0 && ` · ${summary.from_capture} from the captured sidecar of their run`}
        {summary.actual_from_game_logs > 0 &&
          ` · ${summary.actual_from_game_logs} actual starter${summary.actual_from_game_logs === 1 ? "" : "s"} from the game logs`}
      </div>
      <div className="mono dim2" style={{ fontSize: 9, marginTop: 4 }}>
        Who started: {recap.conventions.actual_starter}
      </div>
      <table className="mono tnum" style={{ marginTop: 12, fontSize: 12, borderCollapse: "collapse" }} aria-label="Starter predictor accuracy">
        <thead>
          <tr className="dim" style={{ fontSize: 9 }}>
            <th scope="col" style={{ textAlign: "left", fontWeight: 400, paddingRight: 24 }}>Predictor</th>
            <th scope="col" style={{ textAlign: "right", fontWeight: 400, paddingRight: 16 }}>Right</th>
            <th scope="col" style={{ textAlign: "right", fontWeight: 400 }}>Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {PREDICTORS.map(([label, key]) => (
            <PredictorRow key={key} label={label} accuracy={summary[key]} />
          ))}
        </tbody>
      </table>
      {misses.length > 0 && <MissTable misses={misses} />}
      {misses.length === 0 && summary.scored > 0 && (
        <div className="dim" style={{ fontSize: 12, marginTop: 12 }}>Every predictor named every scored starter.</div>
      )}
      {unscored.size > 0 && (
        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 4 }} aria-label="Unscored starter slots">
          {[...unscored].map(([status, teams]) => (
            <div key={status} data-starter-status={status} style={{ fontSize: 12 }} title={recapStarterStatusMessage(status)}>
              <span style={{ color: "var(--ink-4)" }}>{recapStarterStatusShort(status)} ({teams.length}):</span>{" "}
              <span className="mono">{teams.join(", ")}</span>
              <div className="dim2" style={{ fontSize: 10 }}>{recapStarterStatusMessage(status)}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function PredictorRow({ label, accuracy }: { label: string; accuracy: PredictorAccuracy }) {
  return (
    <tr>
      <th scope="row" className="dim" style={{ textAlign: "left", fontWeight: 400, padding: "3px 24px 3px 0" }}>{label}</th>
      <td style={{ textAlign: "right", paddingRight: 16 }}>{accuracy.correct} / {accuracy.count}</td>
      <td style={{ textAlign: "right" }}>{percent(accuracy.accuracy ?? null)}</td>
    </tr>
  );
}

function MissTable({ misses }: { misses: Slot[] }) {
  return (
    <div className="market-table-scroll" role="region" aria-label="Starter misses" tabIndex={0} style={{ marginTop: 14 }}>
      <table className="recap-table mono">
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
              <th scope="row">{team}</th>
              <td>
                <Pick name={side.assumed_name} id={side.assumed_id} correct={side.assumed_correct} />
                <div className="dim2" style={{ fontSize: 9 }}>
                  {side.assumed_source}
                  {side.assumed_as_of && ` · ${formatTimestamp(side.assumed_as_of)}`}
                  {side.record_source && ` · ${recapRecordSourceLabel(side.record_source)}`}
                </div>
              </td>
              <td><Pick name={side.nflverse_name} id={side.nflverse_id} correct={side.nflverse_correct} /></td>
              <td><Pick name={side.carried_forward_name} id={side.carried_forward_id} correct={side.carry_forward_correct} /></td>
              <td>
                {side.actual_name ?? side.actual_id}
                {side.actual_source && (
                  <div className="dim2" style={{ fontSize: 9 }}>{recapActualSourceLabel(side.actual_source)}</div>
                )}
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
    <span data-starter-pick={mark} style={{ color: correct === false ? "var(--neg)" : undefined }}>
      {name ?? id} <span className="dim2" style={{ fontSize: 9 }}>({mark})</span>
    </span>
  );
}

type Column = [string, (game: RecapGame) => ReactNode];

const FAMILY_COLUMNS: Record<RecapFamily, Column[]> = {
  win: [
    ["Model P(home)", (game) => (
      <WeeklyComponentValue
        label="Win"
        status={game.win.status}
        usable={game.win.status === "available" || game.win.status === "outcome_tie"}
        value={game.win.home_win_prob}
        format={(value) => `${(value * 100).toFixed(1)}%`}
        statusMessage={recapForecastStatusMessage("win", game.win.status)}
      />
    )],
    ["Closing P(home)", (game) => (
      <WeeklyComponentValue
        label="Closing win probability"
        status={game.closing_market.spread_status}
        usable={game.closing_market.spread_status === "available"}
        value={game.closing_market.home_win_prob}
        format={(value) => `${(value * 100).toFixed(1)}%`}
        statusMessage={recapClosingLineMessage(game.closing_market.spread_status)}
      />
    )],
    ["Model Brier", (game) => modelError(game.win.status, game.win.model_brier, (value) => value.toFixed(3))],
    ["Market Brier", (game) =>
      game.win.status === "outcome_tie"
        ? statusText("outcome_tie", recapForecastStatusMessage("win", "outcome_tie"), "Tied game")
        : closingError(game.closing_market.spread_status, game.win.market_brier, (value) => value.toFixed(3))],
    ["Closer", (game) => closer(game.win.closer, game.win.status, game.closing_market.spread_status, "win")],
  ],
  spread: [
    ["Model margin", (game) => modelValue("spread", game.spread.status, game.spread.value, signedOne)],
    ["Line at forecast", (game) => (
      <WeeklyComponentValue
        label="Spread at forecast"
        status={game.market_at_forecast.status}
        usable={game.market_at_forecast.status === "available"}
        value={game.market_at_forecast.home_margin}
        format={signedOne}
        statusMessage={recapMarketAtForecastMessage(game.market_at_forecast.status)}
      />
    )],
    ["Closing margin", (game) => closingValue(game.closing_market.spread_status, game.closing_market.home_margin, signedOne)],
    ["Model error", (game) => modelError(game.spread.status, game.spread.model_error, signedOne)],
    ["Market error", (game) => closingError(game.closing_market.spread_status, game.spread.market_error, signedOne)],
    ["Closer", (game) => closer(game.spread.closer, game.spread.status, game.closing_market.spread_status, "spread")],
  ],
  total: [
    ["Model total", (game) => modelValue("total", game.total.status, game.total.value, one)],
    ["Line at forecast", (game) => (
      <WeeklyComponentValue
        label="Total at forecast"
        status={game.market_at_forecast.status}
        usable={game.market_at_forecast.status === "available"}
        value={game.market_at_forecast.total}
        format={one}
        statusMessage={recapMarketAtForecastMessage(game.market_at_forecast.status)}
      />
    )],
    ["Closing total", (game) => closingValue(game.closing_market.total_status, game.closing_market.total, one)],
    ["Model error", (game) => modelError(game.total.status, game.total.model_error, signedOne)],
    ["Market error", (game) => closingError(game.closing_market.total_status, game.total.market_error, signedOne)],
    ["Closer", (game) => closer(game.total.closer, game.total.status, game.closing_market.total_status, "total")],
  ],
  projected_score: [
    ["Model score", (game) =>
      game.projected_score.status === "available" &&
      game.projected_score.away_score != null &&
      game.projected_score.home_score != null
        ? `${one(game.projected_score.away_score)} – ${one(game.projected_score.home_score)}`
        : statusText(
            game.projected_score.status,
            recapForecastStatusMessage("projected_score", game.projected_score.status),
          )],
    ["Market implied", (game) => {
      const { implied_away_score: away, implied_home_score: home } = game.closing_market;
      return away != null && home != null
        ? `${one(away)} – ${one(home)}`
        : statusText("closing_line_missing", recapClosingLineMessage("closing_line_missing"));
    }],
    ["Model error / team", (game) =>
      modelError(game.projected_score.status, game.projected_score.model_abs_error, two)],
    ["Market error / team", (game) => {
      const missing = game.closing_market.spread_status !== "available" || game.closing_market.total_status !== "available";
      return closingError(missing ? "closing_line_missing" : "available", game.projected_score.market_abs_error, two);
    }],
    ["Closer", (game) => {
      const missing = game.closing_market.spread_status !== "available" || game.closing_market.total_status !== "available";
      return closer(
        game.projected_score.closer,
        game.projected_score.status,
        missing ? "closing_line_missing" : "available",
        "projected_score",
      );
    }],
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

function closingValue(
  status: components["schemas"]["ClosingLineStatus"],
  value: number | null | undefined,
  format: (value: number) => string,
) {
  return (
    <WeeklyComponentValue
      label="Closing line"
      status={status}
      usable={status === "available"}
      value={value}
      format={format}
      statusMessage={recapClosingLineMessage(status)}
    />
  );
}

/** A model error cell: the value, or the family status in short form. */
function modelError(status: ForecastStatus, value: number | null | undefined, format: (value: number) => string) {
  if (status === "available" && value != null) return <span>{format(value)}</span>;
  return statusText(status, recapForecastStatusShort(status), recapForecastStatusShort(status));
}

function closingError(
  status: components["schemas"]["ClosingLineStatus"],
  value: number | null | undefined,
  format: (value: number) => string,
) {
  if (status === "available" && value != null) return <span>{format(value)}</span>;
  return statusText(status, recapClosingLineMessage(status));
}

function closer(
  value: Closer | null | undefined,
  status: ForecastStatus,
  closingStatus: components["schemas"]["ClosingLineStatus"],
  family: RecapFamily,
) {
  if (value) {
    const color = value === "model" ? "var(--pos)" : value === "market" ? "var(--ink-2)" : "var(--ink-3)";
    return <span style={{ color }}>{value === "model" ? "Model" : value === "market" ? "Market" : "Tie"}</span>;
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
      data-weekly-status={status}
      data-weekly-value-state="unavailable"
      title={message}
      aria-label={message}
      style={{ color: "var(--ink-4)" }}
    >
      {text}
    </span>
  );
}

function Toggle({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return (
    <button type="button" aria-pressed={active} onClick={onClick} style={{
      padding: "7px 12px", border: "1px solid var(--line-soft)", borderRadius: 4,
      background: active ? "var(--bg-2)" : "transparent", color: active ? "var(--ink)" : "var(--ink-3)",
      fontFamily: "var(--f-sans)", fontSize: 10, cursor: "pointer",
    }}>{children}</button>
  );
}

function one(value: number) { return value.toFixed(1); }
function two(value: number) { return value.toFixed(2); }
function signedOne(value: number) { return `${value > 0 ? "+" : ""}${value.toFixed(1)}`; }
function percent(value: number | null) { return value == null ? "—" : `${(value * 100).toFixed(1)}%`; }
function fixed(value: number | null, digits: number) { return value == null ? "—" : value.toFixed(digits); }
function signedFixed(value: number | null, digits: number) {
  return value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}
function formatTimestamp(value: string) { return new Date(value).toLocaleString(); }
function formatKickoff(value: string) {
  return new Date(value).toLocaleString(undefined, {
    weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit",
  });
}
