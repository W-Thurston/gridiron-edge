import { Fragment } from "react";
import type { KeyboardEvent } from "react";

import { SCOPE_PANEL_ID } from "./pageConstants";
import type { Scope } from "./pageConstants";

const SCOPES: { value: Scope; label: string; sublabel: string }[] = [
  { value: "week", label: "Week", sublabel: "Live forecasts" },
  { value: "season", label: "Season", sublabel: "Live forecasts" },
  { value: "backtest", label: "Backtest", sublabel: "Historical" },
];

/** Week and Season (live forecasts), a divider, then Backtest (historical). Arrow keys move between tabs. */
export function ScopeTabs({ value, onChange }: { value: Scope; onChange: (scope: Scope) => void }) {
  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const index = SCOPES.findIndex((scope) => scope.value === value);
    const next = SCOPES[(index + step + SCOPES.length) % SCOPES.length];
    onChange(next.value);
    document.getElementById(`performance-scope-${next.value}`)?.focus();
  };
  return (
    <div className="pf-scopes" role="tablist" aria-label="Scope" onKeyDown={move}>
      {SCOPES.map((scope) => (
        <Fragment key={scope.value}>
          {scope.value === "backtest" && <span className="pf-scopes-sep" aria-hidden="true" />}
          <button
            type="button"
            role="tab"
            id={`performance-scope-${scope.value}`}
            aria-selected={scope.value === value}
            aria-controls={SCOPE_PANEL_ID}
            tabIndex={scope.value === value ? 0 : -1}
            onClick={() => onChange(scope.value)}
          >
            <b>{scope.label}</b>
            <small>{scope.sublabel}</small>
          </button>
        </Fragment>
      ))}
    </div>
  );
}
