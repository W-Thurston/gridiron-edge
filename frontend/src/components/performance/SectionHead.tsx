import type { ReactNode } from "react";

/** A mono uppercase eyebrow over a 16px heading, with optional controls on the right. */
export function SectionHead({
  eyebrow,
  title,
  description,
  id,
  children,
}: {
  eyebrow: string;
  title: string;
  description?: ReactNode;
  id?: string;
  children?: ReactNode;
}) {
  return (
    <div className="pf-sec-head">
      <div>
        <div className="pf-eyebrow">{eyebrow}</div>
        <h2 id={id}>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {children}
    </div>
  );
}
