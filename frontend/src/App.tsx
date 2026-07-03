import "./App.css";
import { TopNav } from "./components/chrome/TopNav";
import { NavProvider } from "./context/NavContext";
import { Router } from "./Router";

function App() {
  return (
    <NavProvider>
      <div className="hm-frame">
        <TopNav alertCount={4} slipCount={0} />
        <main style={{ padding: 24, flex: 1 }}>
          <Router />
        </main>
      </div>
    </NavProvider>
  );
}

export default App;
