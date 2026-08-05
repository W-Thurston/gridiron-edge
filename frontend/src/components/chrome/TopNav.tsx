import { useNav } from "../../context/NavContext";
import { BellIcon, BetSlipIcon, SearchIcon } from "./icons";
import { useAppState } from "../../context/AppStateContext";
import { useBetSlip } from "../../context/BetSlipContext";

const NAV_ITEMS = [
  { label: "Today", path: "/today" },
  { label: "Games", path: "/games" },
  { label: "Teams", path: "/teams" },
  { label: "Players", path: "/players" },
  { label: "Compare", path: "/compare" },
  { label: "Lines", path: "/lines" },
  { label: "Live", path: "/live" },
  { label: "News", path: "/news" },
  { label: "Tools", path: "/tools" },
  { label: "My Bets", path: "/mybets" },
] as const;


export function TopNav() {
  const { state } = useAppState();
  const { legs } = useBetSlip();
  const alertCount = state.alerts;
  const slipCount = legs.length;
  const { route, navigate } = useNav();

  return (
    <nav
      style={{
        height: 56,
        background: "var(--bg)",
        borderBottom: "1px solid var(--line-soft)",
        display: "flex",
        alignItems: "center",
        padding: "0 24px",
        gap: 24,
      }}
    >
      {/* Left: Logo lockup */}
      <button
        type="button"
        onClick={() => navigate("/today")}
        aria-label="Go to today's dashboard"
        style={{
          background: "transparent",
          border: "none",
          padding: 0,
          cursor: "pointer",
          font: "inherit",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            width: 18,
            height: 18,
            background: "var(--pos)",
            borderRadius: 2,
          }}
        />
        <span
          style={{
            fontWeight: 600,
            fontSize: 16,
            letterSpacing: "-0.02em",
          }}
        >
          Gridiron Edge
        </span>
        <span
          className="mono dim2"
          style={{ fontSize: 10.5, marginLeft: 4 }}
        >
          v1.0
        </span>
      </button>

      {/* Center: Nav items */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 18,
          flex: 1,
          marginLeft: 24,
        }}
      >
        {NAV_ITEMS.map((item) => {
          const isActive = route.path.startsWith(item.path);
          return (
            <NavItem
              key={item.path}
              label={item.label}
              active={isActive}
              onClick={() => navigate(item.path)}
            />
          );
        })}
      </div>

      {/* Right: Search + notifications + bet slip + avatar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <SearchBox />
        <IconButton label={`${alertCount} unread notifications`}>
          <BellIcon />
          {alertCount > 0 && <Badge color="var(--neg)">{alertCount}</Badge>}
        </IconButton>
        <BetSlipButton
          count={slipCount}
          onClick={() => navigate("/betslip")}
        />
        <Avatar initials="RG" onClick={() => navigate("/settings")} />
      </div>
    </nav>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function NavItem({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        background: "transparent",
        border: "none",
        padding: 0,
        paddingBottom: 4,
        cursor: "pointer",
        font: "inherit",
        fontSize: 13,
        fontWeight: active ? 500 : 400,
        color: active ? "var(--ink)" : "var(--ink-3)",
        borderBottom: active ? "2px solid var(--pos)" : "2px solid transparent",
        transition: "color 90ms ease",
      }}
    >
      {label}
    </button>
  );
}

function SearchBox() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        minWidth: 180,
        padding: "6px 10px",
        background: "var(--bg-1)",
        border: "1px solid var(--line-soft)",
        borderRadius: 5,
        color: "var(--ink-3)",
      }}
    >
      <SearchIcon />
      <input
        type="text"
        placeholder="Search"
        style={{
          background: "transparent",
          border: "none",
          outline: "none",
          color: "var(--ink)",
          fontSize: 13,
          fontFamily: "inherit",
          flex: 1,
          padding: 0,
        }}
      />
      <span
        className="mono"
        style={{
          fontSize: 10,
          color: "var(--ink-4)",
          padding: "1px 4px",
          background: "var(--bg-2)",
          borderRadius: 3,
        }}
      >
        ⌘K
      </span>
    </div>
  );
}

function IconButton({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      style={{
        background: "transparent",
        border: "none",
        padding: 0,
        cursor: "pointer",
        font: "inherit",
        position: "relative",
        width: 32,
        height: 32,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--ink-3)",
        transition: "color 90ms ease",
      }}
    >
      {children}
    </button>
  );
}

function Badge({
  children,
  color = "var(--pos)",
}: {
  children: React.ReactNode;
  color?: string;
}) {
  return (
    <span
      className="mono"
      style={{
        position: "absolute",
        top: -2,
        right: -2,
        fontSize: 9,
        fontWeight: 600,
        color: "var(--bg)",
        background: color,
        padding: "1px 4px",
        borderRadius: 8,
        minWidth: 14,
        textAlign: "center",
      }}
    >
      {children}
    </span>
  );
}

function BetSlipButton({
  count,
  onClick,
}: {
  count: number;
  onClick: () => void;
}) {
  const hasLegs = count > 0;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Open bet slip${count > 0 ? ` (${count} legs)` : ""}`}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 10px",
        background: hasLegs
          ? "color-mix(in oklab, var(--pos) 15%, transparent)"
          : "var(--bg-1)",
        border: `1px solid ${hasLegs ? "var(--pos-dim)" : "var(--line-soft)"}`,
        borderRadius: 5,
        color: hasLegs ? "var(--pos)" : "var(--ink-2)",
        fontSize: 13,
        cursor: "pointer",
        font: "inherit",
        transition: "background 90ms ease",
      }}
    >
      <BetSlipIcon />
      <span>Bet slip</span>
      {count > 0 && (
        <span
          className="mono"
          style={{
            fontSize: 11,
            fontWeight: 500,
            padding: "1px 5px",
            background: "var(--pos)",
            color: "var(--bg)",
            borderRadius: 3,
          }}
        >
          {count}
        </span>
      )}
    </button>
  );
}

function Avatar({
  initials,
  onClick,
}: {
  initials: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Go to settings"
      style={{
        background: "var(--bg-3)",
        border: "none",
        padding: 0,
        cursor: "pointer",
        font: "inherit",
        width: 30,
        height: 30,
        borderRadius: "50%",
        color: "var(--ink-2)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {initials}
    </button>
  );
}
