import type { ReactNode } from "react";

/** The panel chrome every readout shares: a hairline box with a stencil label. */
export function Panel({
  label,
  meta,
  children,
  bodyClassName = "px-4 py-3",
}: {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
}) {
  return (
    <section className="border border-line bg-panel">
      <header className="flex items-baseline justify-between gap-3 border-b border-line px-4 py-2">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.22em] text-dim">{label}</h2>
        {meta ? <div className="font-mono text-[10px] tracking-wide text-faint">{meta}</div> : null}
      </header>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

export type Tone = "neutral" | "caller" | "analyst" | "reject" | "live" | "quiet";

const TONES: Record<Tone, string> = {
  neutral: "border-line bg-raise text-dim",
  caller: "border-caller/30 bg-caller/10 text-caller",
  analyst: "border-analyst/30 bg-analyst/10 text-analyst",
  reject: "border-reject/35 bg-reject/10 text-reject",
  live: "border-live/40 bg-live/15 text-live",
  quiet: "border-transparent bg-transparent text-faint",
};

export function Chip({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/** A 0–1 bar. Confidence and urgency are both read at a glance, never parsed. */
export function Meter({ value, tone = "neutral" }: { value: number; tone?: Tone }) {
  const width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
  const fill =
    tone === "reject"
      ? "bg-reject"
      : tone === "analyst"
        ? "bg-analyst"
        : tone === "live"
          ? "bg-live"
          : "bg-caller";
  return (
    <div className="h-1 w-full overflow-hidden rounded-sm bg-raise">
      <div className={`h-full ${fill} transition-[width] duration-300`} style={{ width }} />
    </div>
  );
}

/** One labelled value in a readout column. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{label}</span>
      <span className="min-w-0 truncate text-right text-[13px] text-text">{children}</span>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-1 text-[13px] leading-relaxed text-faint">{children}</p>;
}
