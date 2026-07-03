import { ChevronRightIcon } from "./icons";

export type BreadcrumbItem = {
  label: string;
  /** Path to navigate to when clicked. If undefined, the entry is not clickable. */
  path?: string;
};

type BreadcrumbProps = {
  items: BreadcrumbItem[];
};

export function Breadcrumb({ items }: BreadcrumbProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        fontSize: 12,
        color: "var(--ink-3)",
      }}
    >
      {items.map((item, idx) => {
        const isLast = idx === items.length - 1;
        const isClickable = !isLast && item.path;

        return (
          <span
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span
              style={{
                color: isLast ? "var(--ink)" : "var(--ink-3)",
                cursor: isClickable ? "pointer" : "default",
                transition: "color 90ms ease",
              }}
            >
              {item.label}
            </span>
            {!isLast && (
              <span style={{ color: "var(--ink-4)", display: "flex" }}>
                <ChevronRightIcon />
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
