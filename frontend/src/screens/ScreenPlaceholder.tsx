export function ScreenPlaceholder({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="hm-card" style={{ padding: 24, maxWidth: 720 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 8 }}>
        {subtitle}
      </div>
      <div style={{ fontSize: 20 }}>{title}</div>
      <div className="mono tnum dim" style={{ fontSize: 12, marginTop: 12 }}>
        Screen stub — populated during Tier 2.
      </div>
    </div>
  );
}
