import type { ReactNode } from "react";

/** Methodology and artifact identities, collapsed at the bottom of a scope. */
export function MethodDetails({ children }: { children: ReactNode }) {
  return (
    <details className="pf-method" data-method-details>
      <summary>Method and provenance</summary>
      <div className="pf-method-body">{children}</div>
    </details>
  );
}
