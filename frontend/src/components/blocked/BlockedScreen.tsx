import { BlockedField } from "../field-status/BlockedField";

type BlockedScreenProps = {
  /** Screen title, e.g. "Line Shopping". */
  title: string;
  /** One-line description of what the screen will show. */
  description: string;
  /** Blocker slug (matches Blocker registry from api/meta.py). */
  blocker: string;
  /** Stable semantic roadmap reference for the blocked capability. */
  roadmap: string;
  /** Bulleted list of what needs to ship before this screen populates. */
  requirements: string[];
};

export function BlockedScreen({
  title,
  description,
  blocker,
  roadmap,
  requirements,
}: BlockedScreenProps) {
  return (
    <div
      className="hm-card"
      style={{
        padding: "48px 32px",
        maxWidth: 720,
        margin: "48px auto 0",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 12,
        }}
      >
        <h1
          className="serif"
          style={{
            fontSize: 32,
            fontWeight: 400,
            letterSpacing: "-0.01em",
            margin: 0,
          }}
        >
          {title}
        </h1>
        <BlockedField blocker={blocker} roadmap={roadmap} placeholder="" />
      </div>

      <p
        style={{
          fontSize: 15,
          color: "var(--ink-2)",
          marginBottom: 32,
          lineHeight: 1.5,
        }}
      >
        {description}
      </p>

      <div
        className="upper dim"
        style={{ fontSize: 10, marginBottom: 12 }}
      >
        What needs to happen
      </div>

      <ul
        style={{
          margin: 0,
          paddingLeft: 20,
          color: "var(--ink-2)",
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        {requirements.map((req, i) => (
          <li key={i} style={{ marginBottom: 6 }}>
            {req}
          </li>
        ))}
      </ul>

      <div
        style={{
          marginTop: 40,
          paddingTop: 20,
          borderTop: "1px solid var(--line-soft)",
        }}
      >
        <div
          className="upper dim2"
          style={{ fontSize: 9, marginBottom: 4 }}
        >
          Tracking
        </div>
        <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>
          Blocker: {blocker}
          <br />
          Roadmap: {roadmap}
        </div>
      </div>
    </div>
  );
}
