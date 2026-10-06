export type Count = { label: string; value: number | string; dim?: boolean };

/** Number-over-label cells in one bordered strip. */
export function CountsStrip({ label, counts }: { label: string; counts: Count[] }) {
  return (
    <ul className="pf-counts" aria-label={label}>
      {counts.map((count) => (
        <li key={count.label} data-count={count.label}>
          <b className={count.dim ? "pf-dim" : undefined}>{count.value}</b>
          <small>{count.label}</small>
        </li>
      ))}
    </ul>
  );
}
