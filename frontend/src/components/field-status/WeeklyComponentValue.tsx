type WeeklyComponentValueProps = {
  label: string;
  status: string;
  usable: boolean;
  value: number | null | undefined;
  format: (value: number) => string;
  statusMessage: string;
};

export function WeeklyComponentValue({
  label,
  status,
  usable,
  value,
  format,
  statusMessage,
}: WeeklyComponentValueProps) {
  if (usable && value != null) {
    return (
      <span
        className="mono tnum"
        data-weekly-status={status}
        title={statusMessage}
      >
        {format(value)}
      </span>
    );
  }

  const inconsistent = usable && value == null;
  const message = inconsistent
    ? `${label} has status ${status} but no value.`
    : statusMessage;

  return (
    <span
      className="mono"
      data-weekly-status={status}
      data-weekly-value-state={inconsistent ? "inconsistent" : "unavailable"}
      title={message}
      aria-label={message}
      style={{ color: inconsistent ? "var(--warn)" : "var(--ink-4)" }}
    >
      {inconsistent ? "Value unavailable" : statusMessage}
    </span>
  );
}
