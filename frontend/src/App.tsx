import { QueryClientProvider } from "@tanstack/react-query";
import "./App.css";
import { queryClient } from "./api/queryClient";
import { TopNav } from "./components/chrome/TopNav";
import { AppStateProvider } from "./context/AppStateContext";
import { BetSlipProvider } from "./context/BetSlipContext";
import { NavProvider } from "./context/NavContext";
import { Router } from "./Router";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppStateProvider>
        <BetSlipProvider>
          <NavProvider>
            <div className="hm-frame">
              <TopNav />
              <main style={{ padding: 24, flex: 1 }}>
                <Router />
              </main>
            </div>
          </NavProvider>
        </BetSlipProvider>
      </AppStateProvider>
    </QueryClientProvider>
  );
}

export default App;
