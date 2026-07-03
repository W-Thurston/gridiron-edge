import "./App.css";
import { TopNav } from "./components/chrome/TopNav";
import { AppStateProvider } from "./context/AppStateContext";
import { BetSlipProvider } from "./context/BetSlipContext";
import { NavProvider } from "./context/NavContext";
import { Router } from "./Router";

function App() {
  return (
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
  );
}

export default App;
