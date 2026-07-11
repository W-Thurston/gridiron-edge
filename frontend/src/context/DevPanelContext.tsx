import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type DevPanelState = {
  highlightPending: boolean;
};

type DevPanelContextValue = {
  state: DevPanelState;
  setState: (partial: Partial<DevPanelState>) => void;
};

const DevPanelContext = createContext<DevPanelContextValue | undefined>(
  undefined,
);

const STORAGE_KEY = "hm-dev-panel";

const DEFAULT_STATE: DevPanelState = {
  highlightPending: false,
};

function loadInitialState(): DevPanelState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<DevPanelState>;
      return { ...DEFAULT_STATE, ...parsed };
    }
  } catch {
    // Ignore parse errors.
  }
  return DEFAULT_STATE;
}

export function DevPanelProvider({ children }: { children: ReactNode }) {
  const [state, setStateInternal] = useState<DevPanelState>(loadInitialState);

  // Persist to localStorage on every change.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Ignore quota errors.
    }
  }, [state]);

  const setState = (partial: Partial<DevPanelState>) => {
    setStateInternal((prev) => ({ ...prev, ...partial }));
  };

  return (
    <DevPanelContext.Provider value={{ state, setState }}>
      {children}
    </DevPanelContext.Provider>
  );
}

export function useDevPanel(): DevPanelContextValue {
  const ctx = useContext(DevPanelContext);
  if (!ctx) {
    throw new Error("useDevPanel must be used inside a DevPanelProvider");
  }
  return ctx;
}
