import "./App.css";

function App() {
  return (
    <div className="hm-frame">
      <header
        style={{
          padding: "20px 24px",
          borderBottom: "1px solid var(--line-soft)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 18,
              height: 18,
              background: "var(--pos)",
              borderRadius: 2,
            }}
          />
          <span style={{ fontWeight: 600, fontSize: 16, letterSpacing: "-0.02em" }}>
            Gridiron Edge
          </span>
          <span className="mono dim2" style={{ fontSize: 10.5 }}>
            v4.2
          </span>
        </div>
      </header>

      <main style={{ padding: 24 }}>
        <div className="hm-card" style={{ padding: 24, maxWidth: 480 }}>
          <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
            Design Tokens Loaded
          </div>
          <div style={{ fontSize: 20, marginBottom: 16 }}>
            Ready for W9 Tier 1 Substep 1c
          </div>
          <div className="mono tnum" style={{ fontSize: 12 }}>
            <div>
              Font: <span className="pos">Geist Mono</span>
            </div>
            <div>
              Positive semantic: <span className="pos">+14.2%</span>
            </div>
            <div>
              Negative semantic: <span className="neg">-5.7%</span>
            </div>
            <div className="dim">Dim text</div>
            <div className="dim2">Dimmer text</div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
