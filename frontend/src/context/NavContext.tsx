import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

/** Route params — used for ?gameId=, ?team=, ?propId= style routes. */
export type RouteParams = Record<string, string>;

/** Current route state. */
export type Route = {
  path: string;
  params: RouteParams;
};

/** Public shape of the nav context. */
type NavContextValue = {
  route: Route;
  navigate: (path: string, params?: RouteParams) => void;
};

const NavContext = createContext<NavContextValue | undefined>(undefined);

const STORAGE_KEY = "hm-route";

/**
 * Parse a hash string ("#/games?gameId=2026_01_KC_LAC") into a Route.
 * Returns null for invalid hashes.
 */
function parseHash(hash: string): Route | null {
  // Strip leading "#/" or "#".
  const cleaned = hash.replace(/^#\/?/, "");
  if (!cleaned) return null;

  const [pathPart, queryPart] = cleaned.split("?");
  const path = "/" + pathPart;

  const params: RouteParams = {};
  if (queryPart) {
    const searchParams = new URLSearchParams(queryPart);
    for (const [key, value] of searchParams.entries()) {
      params[key] = value;
    }
  }

  return { path, params };
}

/**
 * Serialize a Route into a hash string. Reverse of parseHash.
 */
function serializeRoute(route: Route): string {
  const pathPart = route.path.replace(/^\//, "");
  const paramEntries = Object.entries(route.params);

  if (paramEntries.length === 0) {
    return `#/${pathPart}`;
  }

  const query = new URLSearchParams(route.params).toString();
  return `#/${pathPart}?${query}`;
}

/** Read the initial route from window.location.hash, sessionStorage, or default. */
function getInitialRoute(): Route {
  // 1. Try the URL hash first (user pasted a link).
  const fromHash = parseHash(window.location.hash);
  if (fromHash) return fromHash;

  // 2. Try sessionStorage (user refreshed).
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Route;
      if (parsed.path) return parsed;
    }
  } catch {
    // Ignore parse errors.
  }

  // 3. Default.
  return { path: "/today", params: {} };
}

export function NavProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(getInitialRoute);

  // Sync route → URL hash and sessionStorage.
  useEffect(() => {
    const nextHash = serializeRoute(route);
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, "", nextHash);
    }
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(route));
    } catch {
      // Ignore quota errors.
    }
  }, [route]);

  // Sync URL hash → route (when user uses back/forward buttons).
  useEffect(() => {
    const onHashChange = () => {
      const fromHash = parseHash(window.location.hash);
      if (fromHash) {
        setRoute(fromHash);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((path: string, params: RouteParams = {}) => {
    setRoute({ path, params });
  }, []);

  return (
    <NavContext.Provider value={{ route, navigate }}>
      {children}
    </NavContext.Provider>
  );
}

/** Hook to access the current nav state. */
export function useNav(): NavContextValue {
  const ctx = useContext(NavContext);
  if (!ctx) {
    throw new Error("useNav must be used inside a NavProvider");
  }
  return ctx;
}
