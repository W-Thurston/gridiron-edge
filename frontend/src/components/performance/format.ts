/** Number formats shared by the Performance page; every one renders a missing value as an em dash. */
export function fixed(value: number | null | undefined, digits: number): string {
  return value == null ? "—" : value.toFixed(digits);
}
export function percent(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
export function signedFixed(value: number | null | undefined, digits: number): string {
  return value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}
export function signedCount(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString()}`;
}
export function one(value: number): string { return value.toFixed(1); }
export function two(value: number): string { return value.toFixed(2); }
export function signedOne(value: number): string { return `${value > 0 ? "+" : ""}${value.toFixed(1)}`; }
export function formatTimestamp(value: string): string { return new Date(value).toLocaleString(); }
export function formatShortTimestamp(value: string): string {
  return new Date(value).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}
export function formatKickoff(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit",
  });
}
