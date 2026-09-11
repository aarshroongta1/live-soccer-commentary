import type { StateEvent } from "@/lib/events";

/**
 * What the system believes the match is, read off the screen and nowhere else.
 * Every number here came from the board reader or the caller, so it is also a
 * readout of how well perception is doing.
 */
export function Scoreboard({ state }: { state: StateEvent | null }) {
  const home = state?.home ?? "—";
  const away = state?.away ?? "—";
  const clock = state?.clock ?? "--:--";
  const period = state ? periodLabel(state.period) : "—";
  const inReplay = state?.inReplay ?? false;

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border border-line bg-panel px-4 py-3 sm:px-6">
      {/* On a phone the clock drops to its own line rather than squeezing the
          team names down to nothing. */}
      <div className="flex w-full min-w-0 items-center gap-3 sm:w-auto sm:flex-1 sm:gap-6">
        <Team name={home} align="right" hasBall={state?.possession === "home"} />
        {/* Dashes, not zeroes: before the board has been read the system does
            not know the score, and nil-nil would be a claim. */}
        <div
          className={`tnum shrink-0 font-mono text-[34px] leading-none tracking-tight sm:text-[40px] ${
            state ? "text-text" : "text-faint"
          }`}
        >
          {state ? state.homeScore : "–"}
          <span className="px-2 text-faint">–</span>
          {state ? state.awayScore : "–"}
        </div>
        <Team name={away} align="left" hasBall={state?.possession === "away"} />
      </div>

      <div className="flex w-full items-center justify-between gap-4 sm:w-auto sm:justify-end sm:gap-6">
        <div className="text-right">
          <div className="tnum font-mono text-[22px] leading-none text-text">{clock}</div>
          <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">{period}</div>
        </div>
        <ReplayLamp on={inReplay} />
      </div>
    </div>
  );
}

function Team({ name, align, hasBall }: { name: string; align: "left" | "right"; hasBall: boolean }) {
  const order = align === "right" ? "flex-row-reverse text-right" : "flex-row text-left";
  return (
    <div className={`flex min-w-0 flex-1 items-center gap-2 ${order}`}>
      <span className="min-w-0 truncate text-[15px] font-medium uppercase tracking-[0.09em] text-text sm:text-[17px]">
        {name}
      </span>
      <span
        className={`h-1.5 w-1.5 shrink-0 rounded-full ${hasBall ? "bg-caller" : "bg-line"}`}
        title={hasBall ? "in possession" : undefined}
      />
    </div>
  );
}

/**
 * The score bug vanishing is how the system knows it is in a replay, and a
 * replay is the single most common way a commentator embarrasses itself — so
 * the lamp stays on screen, dark, rather than appearing only when it fires.
 */
function ReplayLamp({ on }: { on: boolean }) {
  return (
    <div
      className={`flex items-center gap-2 rounded-sm border px-2 py-1.5 transition-colors ${
        on ? "border-live/50 bg-live/15" : "border-line bg-raise"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-live live-dot" : "bg-faint/40"}`} />
      <span
        className={`font-mono text-[10px] uppercase tracking-[0.2em] ${on ? "text-live" : "text-faint/70"}`}
      >
        replay
      </span>
    </div>
  );
}

function periodLabel(period: number): string {
  if (period === 1) return "first half";
  if (period === 2) return "second half";
  return `period ${period}`;
}
