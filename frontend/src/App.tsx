import { QueryClientProvider } from "@tanstack/react-query";
import "./App.css";
import { queryClient } from "./api/queryClient";
import { TopNav } from "./components/chrome/TopNav";
import { OfflineBanner } from "./components/error/OfflineBanner";
import { AppStateProvider } from "./context/AppStateContext";
import { BetSlipProvider } from "./context/BetSlipContext";
import { DevPanelProvider } from "./context/DevPanelContext";
import { NavProvider } from "./context/NavContext";
import { Router } from "./Router";
import { DevPanel } from "./components/dev/DevPanel";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppStateProvider>
        <DevPanelProvider>
          <BetSlipProvider>
            <NavProvider>
              <div className="hm-frame">
              <OfflineBanner />
              <TopNav />
              <main style={{ padding: 24, flex: 1 }}>
                <Router />
              </main>
            </div>
            <DevPanel />
            </NavProvider>
          </BetSlipProvider>
        </DevPanelProvider>
      </AppStateProvider>
    </QueryClientProvider>
  );
}

export default App;
