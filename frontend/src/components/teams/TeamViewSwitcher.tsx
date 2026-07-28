import { useNav } from "../../context/NavContext";

type TeamView = "rankings" | "projections";

type TeamViewSwitcherProps = {
  active: TeamView;
};

const OPTIONS: Array<{
  key: TeamView;
  label: string;
  path: string;
}> = [
  {
    key: "rankings",
    label: "Team Rankings",
    path: "/teams",
  },
  {
    key: "projections",
    label: "Playoff Projections",
    path: "/projections",
  },
];

/**
 * Sibling-view navigation between team rankings and playoff projections.
 */
export function TeamViewSwitcher({
  active,
}: TeamViewSwitcherProps) {
  const { navigate } = useNav();

  return (
    <nav
      aria-label="Team analysis views"
      style={{
        display: "flex",
        gap: 4,
        marginBottom: 12,
        padding: 3,
        width: "fit-content",
        border: "1px solid var(--line-soft)",
        borderRadius: 5,
        background: "var(--bg-1)",
      }}
    >
      {OPTIONS.map((option) => {
        const isActive = option.key === active;

        return (
          <button
            key={option.key}
            type="button"
            aria-current={isActive ? "page" : undefined}
            onClick={() => {
              if (!isActive) {
                navigate(option.path);
              }
            }}
            style={{
              padding: "5px 11px",
              border: 0,
              borderRadius: 3,
              background: isActive
                ? "var(--pos)"
                : "transparent",
              color: isActive
                ? "var(--bg)"
                : "var(--ink-3)",
              fontFamily: "var(--f-sans)",
              fontSize: 11,
              cursor: isActive ? "default" : "pointer",
            }}
          >
            {option.label}
          </button>
        );
      })}
    </nav>
  );
}
