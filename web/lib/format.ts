/** Small formatters shared by the scoreboard, the feed and the agent panel. */

/** Video-buffer seconds as the clock a viewer would read, e.g. `52:26`. */
export function videoClock(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  const rest = Math.floor(safe % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

/** Enum values arrive snake_cased; they read better spaced. */
export function label(value: string): string {
  return value.replace(/_/g, " ");
}

export function percent(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

/** Spend is shown to the cent once it matters, and to the tenth of a cent before. */
export function usd(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value >= 1 ? `$${value.toFixed(2)}` : `$${value.toFixed(3)}`;
}

export function secondsAgo(from: number, now: number): string {
  const delta = Math.max(0, Math.round((now - from) / 1000));
  if (delta < 1) return "just now";
  if (delta < 60) return `${delta}s ago`;
  return `${Math.floor(delta / 60)}m ago`;
}
