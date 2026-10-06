import { useId, useState } from "react";
import type { CSSProperties, KeyboardEvent, MouseEvent, ReactNode } from "react";
import { SectionHead } from "./SectionHead";
import { SegmentedControl } from "./SegmentedControl";
import { GROUP_LABELS, MODEL_STYLES } from "./compareData";
import { segments, summarize, tooltipRows } from "./compareMath";
import type { CompareData, CompareMetric, CompareModel, CompareModelGroup, CompareModelId, CompareSource } from "./compareData";

type Props = {
  eyebrow: string;
  title: string;
  source: CompareSource;
  selected: CompareModelId[];
  onSelectedChange: (selected: CompareModelId[]) => void;
  metricKey: string | undefined;
  onMetricChange: (metric: string) => void;
  /** Extra controls beside the metric selector (for example a week / to-date view). */
  controls?: ReactNode;
  height?: number;
};

const GROUPS: CompareModelGroup[] = ["ours", "market", "external"];

/** Champion against other models, one line each, with better always up and a summary table below. */
export function CompareChart({
  eyebrow, title, source, selected, onSelectedChange, metricKey, onMetricChange, controls, height = 220,
}: Props) {
  const headingId = useId();
  if (source.status === "unavailable") {
    return (
      <section className="hm-card pf-card" aria-labelledby={headingId} data-compare-chart="unavailable">
        <SectionHead eyebrow={eyebrow} title={title} id={headingId} />
        <div className="pf-empty" style={{ padding: 0 }} data-weekly-status="not_persisted" data-weekly-value-state="unavailable">
          <h2 style={{ fontSize: 14 }}>{source.reason}</h2>
          <p>{source.detail}</p>
        </div>
      </section>
    );
  }
  const { data, metrics } = source;
  const metric = metrics.find((m) => m.key === metricKey) ?? metrics[0];
  return (
    <section className="hm-card pf-card" aria-labelledby={headingId} data-compare-chart={metric.key}>
      <SectionHead eyebrow={eyebrow} title={title} id={headingId}>
        <div className="pf-seg-group">
          {controls}
          <SegmentedControl
            label="Comparison metric"
            options={metrics.map((m) => ({ value: m.key, label: m.label }))}
            value={metric.key}
            onChange={onMetricChange}
          />
        </div>
      </SectionHead>
      <ComparePlot data={data} metric={metric} selected={selected} onSelectedChange={onSelectedChange} height={height} />
    </section>
  );
}

function ComparePlot({
  data, metric, selected, onSelectedChange, height,
}: {
  data: CompareData;
  metric: CompareMetric;
  selected: CompareModelId[];
  onSelectedChange: (selected: CompareModelId[]) => void;
  height: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState<CompareModelId | null>(null);
  const values = data.values[metric.key] ?? {};
  const available = (id: CompareModelId) => (values[id] ?? []).some((v) => v != null);
  const visible = data.models.filter((m) => selected.includes(m.id) && available(m.id));
  const n = data.periods.length;
  const toggle = (id: CompareModelId) =>
    onSelectedChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);

  return (
    <>
      <div className="pc-chips">
        {GROUPS.map((group) => {
          const models = data.models.filter((m) => m.group === group);
          if (models.length === 0) return null;
          return (
            <div className="pc-group" key={group} role="group" aria-label={`${GROUP_LABELS[group]} models`}>
              <span className="pc-glabel" aria-hidden="true">{GROUP_LABELS[group]}</span>
              {models.map((model) => {
                const ok = available(model.id);
                const on = ok && selected.includes(model.id);
                return (
                  <button
                    key={model.id}
                    type="button"
                    className="pc-chip"
                    data-compare-model={model.id}
                    disabled={!ok}
                    aria-pressed={on}
                    title={ok ? `${model.label} · ${model.source}` : `${model.label} has no ${metric.label} for this view`}
                    onClick={() => toggle(model.id)}
                    onMouseEnter={() => on && setFocus(model.id)}
                    onMouseLeave={() => setFocus(null)}
                    onFocus={() => on && setFocus(model.id)}
                    onBlur={() => setFocus(null)}
                  >
                    <Swatch id={model.id} />
                    {model.label}
                    {!ok && <small>n/a</small>}
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
      {visible.length === 0 || n === 0 ? (
        <div className="pc-none">Select at least one model to compare.</div>
      ) : (
        <>
          <Plot
            data={data} metric={metric} visible={visible} height={height}
            hover={hover} setHover={setHover} focus={focus}
          />
          <SummaryTable data={data} metric={metric} visible={visible} setFocus={setFocus} />
          <div className="pf-note">
            Averages are unweighted means of the period values. Differences from the champion use the periods both
            cover.{metric.note ? ` ${metric.note}` : ""}
          </div>
        </>
      )}
    </>
  );
}

function Swatch({ id }: { id: CompareModelId }) {
  const style = MODEL_STYLES[id];
  return (
    <i
      className={`pc-swatch${style.dash ? " pc-swatch-dash" : ""}`}
      style={{ "--c": style.color } as CSSProperties}
      aria-hidden="true"
    />
  );
}

function Plot({
  data, metric, visible, height, hover, setHover, focus,
}: {
  data: CompareData;
  metric: CompareMetric;
  visible: CompareModel[];
  height: number;
  hover: number | null;
  setHover: (index: number | null) => void;
  focus: CompareModelId | null;
}) {
  const values = data.values[metric.key] ?? {};
  const series = (id: CompareModelId) => values[id] ?? [];
  const n = data.periods.length;
  const drawn = visible.flatMap((m) => series(m.id).filter((v): v is number => v != null));
  let lo = Math.min(...drawn);
  let hi = Math.max(...drawn);
  const pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.05 || 0.01;
  lo -= pad;
  hi += pad;
  const x = (i: number) => (n === 1 ? 500 : (i / (n - 1)) * 1000);
  // Better is always up: for a lower-is-better metric the smallest value sits at the top.
  const y = (v: number) => {
    const t = (v - lo) / (hi - lo);
    return metric.better === "lower" ? t * height : height - t * height;
  };
  const ticks = [0, 1, 2, 3, 4].map((i) => lo + ((hi - lo) * i) / 4);
  const labelEvery = Math.max(1, Math.ceil(n / 12));

  const indexAt = (event: MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || n === 1) return 0;
    return Math.max(0, Math.min(n - 1, Math.round(((event.clientX - rect.left) / rect.width) * (n - 1))));
  };
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      setHover(Math.max(0, Math.min(n - 1, (hover ?? (step > 0 ? -1 : n)) + step)));
    } else if (event.key === "Escape") {
      setHover(null);
    }
  };

  return (
    <div className="pc-chart" style={{ height: height + 26 }}>
      <div className="pc-yaxis" style={{ height }} aria-hidden="true">
        {ticks.map((tick) => <span key={tick} style={{ top: y(tick) }}>{metric.format(tick)}</span>)}
      </div>
      <div
        className="pc-plot"
        tabIndex={0}
        role="group"
        aria-label={`${metric.label} by period. Use the left and right arrow keys to read each period.`}
        onMouseMove={(event) => setHover(indexAt(event))}
        onMouseLeave={() => setHover(null)}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
      >
        <span className="pc-better">{metric.better === "lower" ? "↑ lower is better" : "↑ higher is better"}</span>
        <svg
          viewBox={`0 0 1000 ${height}`}
          preserveAspectRatio="none"
          style={{ width: "100%", height, display: "block", overflow: "visible" }}
          aria-hidden="true"
        >
          {ticks.map((tick) => (
            <line key={tick} x1={0} x2={1000} y1={y(tick)} y2={y(tick)} stroke="var(--line-soft)" vectorEffect="non-scaling-stroke" />
          ))}
          {hover != null && (
            <line x1={x(hover)} x2={x(hover)} y1={0} y2={height} stroke="var(--line)" vectorEffect="non-scaling-stroke" />
          )}
          {visible.map((model) =>
            segments(series(model.id)).map((segment) => (
              <polyline
                key={`${model.id}-${segment[0][0]}`}
                data-compare-line={model.id}
                points={segment.map(([i, v]) => `${x(i)},${y(v)}`).join(" ")}
                fill="none"
                stroke={MODEL_STYLES[model.id].color}
                strokeWidth={model.id === "champion" ? 2.25 : 1.75}
                strokeDasharray={MODEL_STYLES[model.id].dash ? "5 4" : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
                style={{ opacity: focus && focus !== model.id ? 0.18 : 1, transition: "opacity 120ms" }}
              />
            )),
          )}
        </svg>
        {/* A period with no neighbour has no line to draw; mark it so it is not lost. */}
        {visible.flatMap((model) =>
          segments(series(model.id)).filter((segment) => segment.length === 1).map(([[i, v]]) => (
            <i
              key={`${model.id}-solo-${i}`}
              className="pc-dot"
              style={{ left: `${x(i) / 10}%`, top: y(v), background: MODEL_STYLES[model.id].color,
                opacity: focus && focus !== model.id ? 0.18 : 1 }}
            />
          )),
        )}
        {hover != null && visible.map((model) => {
          const v = series(model.id)[hover];
          return v == null ? null : (
            <i key={model.id} className="pc-dot" style={{ left: `${x(hover) / 10}%`, top: y(v), background: MODEL_STYLES[model.id].color }} />
          );
        })}
        {hover != null && (
          <div
            className={`pc-tip${hover > (n - 1) * 0.6 ? " pc-tip-left" : ""}`}
            style={{ left: `${x(hover) / 10}%` }}
            role="tooltip"
            data-compare-tooltip
          >
            <div className="pc-tip-head">{data.periods[hover]} · {metric.label}</div>
            {tooltipRows(data, metric, visible, hover).map(({ model, value }) => (
              <div key={model.id} className="pc-tip-row" data-tooltip-model={model.id}>
                <i style={{ background: MODEL_STYLES[model.id].color }} />
                <span>{model.label}</span>
                <b>{metric.format(value)}</b>
              </div>
            ))}
            {tooltipRows(data, metric, visible, hover).length === 0 && (
              <div className="pc-tip-row"><span className="pf-faint">No selected model has a value</span></div>
            )}
          </div>
        )}
        <div className="pc-xaxis" aria-hidden="true">
          {data.periods.map((label, i) =>
            i % labelEvery === 0 || i === n - 1 ? <span key={label} style={{ left: `${x(i) / 10}%` }}>{label}</span> : null,
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryTable({
  data, metric, visible, setFocus,
}: {
  data: CompareData;
  metric: CompareMetric;
  visible: CompareModel[];
  setFocus: (id: CompareModelId | null) => void;
}) {
  const n = data.periods.length;
  return (
    <div className="pf-scroll">
      <table className="pc-table">
        <caption className="visually-hidden">{metric.label}: each selected model's average against the champion</caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Average {metric.short}</th>
            <th scope="col">vs champion</th>
            <th scope="col">Best in period</th>
            <th scope="col">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {summarize(data, metric, visible).map((row) => {
            const better = row.delta != null && (metric.better === "lower" ? row.delta < 0 : row.delta > 0);
            // Even at display precision, so a rounding-level difference never reads "+0.000 worse".
            const even = row.delta != null && metric.formatDelta(Math.abs(row.delta)) === metric.formatDelta(0);
            return (
              <tr
                key={row.model.id}
                data-compare-row={row.model.id}
                onMouseEnter={() => setFocus(row.model.id)}
                onMouseLeave={() => setFocus(null)}
              >
                <th scope="row">
                  <span><Swatch id={row.model.id} />{row.model.label}<small>{row.model.source}</small></span>
                </th>
                <td>{row.average == null ? "—" : metric.format(row.average)}</td>
                <td className={row.model.id === "champion" || even || row.delta == null ? "pf-faint" : better ? "pc-better-text" : "pc-worse-text"}>
                  {row.model.id === "champion"
                    ? "reference"
                    : row.delta == null
                      ? "No shared periods"
                      : even
                        ? "Even"
                        : `${row.delta > 0 ? "+" : "−"}${metric.formatDelta(Math.abs(row.delta))} ${better ? "better" : "worse"}`}
                </td>
                <td className="pf-muted">best in {row.best} of {row.covered}</td>
                <td className={row.covered === n ? "pf-muted" : "pc-partial"}>
                  {row.covered === n
                    ? "Full"
                    : `${data.periods[row.first]}–${data.periods[row.last]}${row.last - row.first + 1 > row.covered ? ` · ${row.covered} of ${n}` : ""}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
