import "./App.css";
import { Breadcrumb } from "./components/chrome/Breadcrumb";
import { SubNav } from "./components/chrome/SubNav";
import { TopNav } from "./components/chrome/TopNav";

function App() {
  return (
    <div className="hm-frame">
      <TopNav activePath="/today" alertCount={4} slipCount={0} />
      <SubNav
        left={
          <Breadcrumb
            items={[
              { label: "Today", path: "/today" },
              { label: "Week 12" },
            ]}
          />
        }
        right={
          <span className="upper dim" style={{ fontSize: 10 }}>
            Sep 2026 · Week 12
          </span>
        }
      />

      <main style={{ padding: 24, flex: 1 }}>
        <div className="hm-card" style={{ padding: 24, maxWidth: 720 }}>
          <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
            Chrome Components Rendered
          </div>
          <div style={{ fontSize: 20, marginBottom: 16 }}>
            TopNav, SubNav, and Breadcrumb are wired.
          </div>
          <div className="mono tnum dim" style={{ fontSize: 12 }}>
            Next: routing (1d), then three Contexts (1e).
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
