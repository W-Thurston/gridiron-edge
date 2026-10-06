import { useEffect } from "react";
import { useRecaps, useSeasonRecap, useWeeklyRecap } from "../api/hooks";
import type { components } from "../api/schema";
import { ErrorCard } from "../components/error/ErrorCard";
import type { RecapFamily } from "../components/field-status/recapStatus";
import { BacktestScope } from "../components/performance/BacktestScope";
import type { BacktestMarket, BacktestState, BacktestTrend } from "../components/performance/BacktestScope";
import { ScopeTabs } from "../components/performance/ScopeTabs";
import { FAMILIES, SCOPE_PANEL_ID } from "../components/performance/pageConstants";
import type { Scope } from "../components/performance/pageConstants";
import { SeasonNotBuilt, SeasonScope } from "../components/performance/SeasonScope";
import type { SeasonState } from "../components/performance/SeasonScope";
import { WeekScope, WeekStrip } from "../components/performance/WeekScope";
import type { SelectedWeek } from "../components/performance/WeekScope";
import { BACKTEST_DEFAULT_MODELS, SEASON_DEFAULT_MODELS } from "../components/performance/compareData";
import type { CompareModelId } from "../components/performance/compareData";
import { useNav } from "../context/NavContext";
import type { RouteParams } from "../context/NavContext";

type RecapSummary = components["schemas"]["WeeklyRecapSummary"];

const PATH = "/performance";

const DESCRIPTIONS: Record<Scope, string> = {
  week: "How published forecasts did against results and the closing market.",
  season: "How published forecasts did against results and the closing market.",
  backtest: "How the current models would have done on past seasons.",
};

/** Comparison selections belong to one scope; they are dropped when the scope changes. */
const SCOPE_ONLY_PARAMS = ["models", "metric"];

/**
 * Week, Season and Backtest on one page. The scope, the selected week and every sub-selection live
 * in the route params, so a link reproduces the view.
 */
export function Performance() {
  const { route, navigate } = useNav();
  const params = route.params;
  const scope = parseScope(params.scope);

  /** Sub-selections replace the history entry; changing scope or week adds one. */
  const update = (patch: RouteParams) => {
    navigate(PATH, { ...params, ...patch }, { replace: true });
  };
  const chooseScope = (next: Scope) => {
    if (next !== scope) navigate(PATH, { ...withoutScopeOnly(params), scope: next });
  };
  const openWeek = (season: string, week: number) => {
    navigate(PATH, { ...withoutScopeOnly(params), scope: "week", season, week: String(week) });
    window.scrollTo?.(0, 0);
  };

  return (
    <div className="pf-page">
      <div className="pf-top">
        <div>
          <div className="pf-eyebrow">Model performance</div>
          <p>{DESCRIPTIONS[scope]}</p>
        </div>
        <ScopeTabs value={scope} onChange={chooseScope} />
      </div>
      <div id={SCOPE_PANEL_ID} role="tabpanel" aria-labelledby={`performance-scope-${scope}`} className="pf-scope">
        {scope === "week" && (
          <WeekContainer
            params={params}
            onSelectWeek={(summary) => openWeek(summary.season, summary.week)}
            onFamily={(family) => update({ family })}
          />
        )}
        {scope === "season" && (
          <SeasonContainer
            params={params}
            onChange={(next) => update(seasonPatch(next))}
            onOpenWeek={openWeek}
          />
        )}
        {scope === "backtest" && (
          <BacktestScope
            state={backtestState(params)}
            onChange={(next) => update(backtestPatch(next))}
            onOpenScope={chooseScope}
          />
        )}
      </div>
    </div>
  );
}

/** Sends an old route to its scope on this page, replacing the history entry. */
export function PerformanceRedirect({ params }: { params: RouteParams }) {
  const { navigate } = useNav();
  const target = JSON.stringify(params);
  useEffect(() => {
    navigate(PATH, JSON.parse(target) as RouteParams, { replace: true });
  }, [navigate, target]);
  return null;
}

function WeekContainer({
  params, onSelectWeek, onFamily,
}: {
  params: RouteParams;
  onSelectWeek: (summary: RecapSummary) => void;
  onFamily: (family: RecapFamily) => void;
}) {
  const recaps = useRecaps();
  const items = recaps.data?.items ?? [];
  const selected = selectedWeek(params, items);
  const recap = useWeeklyRecap(selected?.season ?? null, selected?.week ?? null);

  if (recaps.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading weekly recaps…</div>;
  }
  if (recaps.error) return <ErrorCard error={recaps.error} onRetry={() => recaps.refetch()} />;
  if (!selected) return <NoRecaps />;

  const strip = <WeekStrip items={items} selected={selected} onSelect={onSelectWeek} />;
  const family = FAMILIES.includes(params.family as RecapFamily) ? (params.family as RecapFamily) : "win";
  return (
    <>
      {!recap.data && (
        <header className="pf-head">
          <div className="pf-head-top">
            <div className="pf-eyebrow">Weekly recap</div>
            {strip}
          </div>
        </header>
      )}
      {recap.isLoading && (
        <div className="hm-card" style={{ padding: 24 }}>Loading the week {selected.week} recap…</div>
      )}
      {recap.error && <ErrorCard error={recap.error} onRetry={() => recap.refetch()} />}
      {recap.data && <WeekScope recap={recap.data} strip={strip} family={family} onFamily={onFamily} />}
    </>
  );
}

function SeasonContainer({
  params, onChange, onOpenWeek,
}: {
  params: RouteParams;
  onChange: (next: Partial<SeasonState>) => void;
  onOpenWeek: (season: string, week: number) => void;
}) {
  const recaps = useRecaps();
  const items = recaps.data?.items ?? [];
  const season = params.season ?? selectedWeek({}, items)?.season ?? null;
  const query = useSeasonRecap(season);

  if (recaps.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading weekly recaps…</div>;
  }
  if (recaps.error) return <ErrorCard error={recaps.error} onRetry={() => recaps.refetch()} />;
  if (!season) return <NoRecaps />;
  if (query.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading the season recap…</div>;
  }
  if (query.error) return <ErrorCard error={query.error} onRetry={() => query.refetch()} />;
  if (!query.data) return <SeasonNotBuilt season={season} />;
  return (
    <SeasonScope
      recap={query.data}
      state={seasonState(params)}
      onChange={onChange}
      onOpenWeek={(week) => onOpenWeek(season, week)}
    />
  );
}

function NoRecaps() {
  return (
    <section className="hm-card pf-empty">
      <div className="pf-eyebrow">Weekly recap</div>
      <h2>No weekly recap has been built yet</h2>
      <p>
        A recap is built once every game of a week has a result:{" "}
        <span className="mono">gridiron evaluate weekly-recap --season S --week W</span>
      </p>
    </section>
  );
}

/** The week named by the route, else the highest (season, week) among the selected recaps. */
function selectedWeek(params: RouteParams, items: RecapSummary[]): SelectedWeek | null {
  const week = Number(params.week);
  if (params.season && Number.isInteger(week) && week >= 1) {
    return { season: params.season, week };
  }
  const latest = [...items].sort(
    (a, b) => a.season.localeCompare(b.season) || a.week - b.week,
  ).at(-1);
  return latest ? { season: latest.season, week: latest.week } : null;
}

function parseScope(value: string | undefined): Scope {
  return value === "season" || value === "backtest" ? value : "week";
}

function withoutScopeOnly(params: RouteParams): RouteParams {
  return Object.fromEntries(Object.entries(params).filter(([key]) => !SCOPE_ONLY_PARAMS.includes(key)));
}

/** A model list survives a reload as a comma list; an empty list is kept as an explicit empty value. */
function parseModels(value: string | undefined, fallback: CompareModelId[]): CompareModelId[] {
  if (value === undefined) return fallback;
  return value.split(",").filter(Boolean) as CompareModelId[];
}

function seasonState(params: RouteParams): SeasonState {
  return {
    games: params.games === "live" ? "live" : "all",
    view: params.view === "to_date" ? "to_date" : "week",
    models: parseModels(params.models, SEASON_DEFAULT_MODELS),
    metric: params.metric,
  };
}

function seasonPatch(next: Partial<SeasonState>): RouteParams {
  const patch: RouteParams = {};
  if (next.games) patch.games = next.games;
  if (next.view) patch.view = next.view;
  if (next.models) patch.models = next.models.join(",");
  if (next.metric) patch.metric = next.metric;
  return patch;
}

function backtestState(params: RouteParams): BacktestState {
  const market: BacktestMarket = params.market === "total" || params.market === "spread" ? params.market : "moneyline";
  const trend: BacktestTrend = params.trend === "accuracy" || params.trend === "units" ? params.trend : "net-wins";
  return { market, trend, models: parseModels(params.models, BACKTEST_DEFAULT_MODELS), metric: params.metric };
}

function backtestPatch(next: Partial<BacktestState>): RouteParams {
  const patch: RouteParams = {};
  if (next.market) patch.market = next.market;
  if (next.trend) patch.trend = next.trend;
  if (next.models) patch.models = next.models.join(",");
  if (next.metric) patch.metric = next.metric;
  return patch;
}
