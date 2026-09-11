"use client";

import { useEffect, useRef, useState } from "react";

import { videoClock } from "@/lib/format";
import type { FeedItem } from "@/lib/store";

/**
 * The transcript: what actually went to air, oldest at the top.
 *
 * A line the director cut appears here too, because part of it was spoken and
 * the viewer heard it. The part that escaped is set normally and the rest is
 * struck through, so the cut is visible as the sentence it interrupted.
 */
export function Feed({ items }: { items: readonly FeedItem[] }) {
  const scroller = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);

  useEffect(() => {
    const node = scroller.current;
    if (!node || !pinned) return;
    node.scrollTop = node.scrollHeight;
  }, [items, pinned]);

  return (
    <section className="flex min-h-0 flex-1 flex-col border border-line bg-panel">
      <header className="flex shrink-0 items-baseline justify-between border-b border-line px-4 py-2">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.22em] text-dim">commentary</h2>
        {!pinned ? (
          <button
            type="button"
            onClick={() => setPinned(true)}
            className="font-mono text-[10px] uppercase tracking-[0.16em] text-caller hover:text-text"
          >
            jump to live
          </button>
        ) : (
          <span className="font-mono text-[10px] tracking-wide text-faint">{items.length} lines</span>
        )}
      </header>

      <div
        ref={scroller}
        onScroll={(event) => {
          const node = event.currentTarget;
          setPinned(node.scrollHeight - node.scrollTop - node.clientHeight < 48);
        }}
        className="min-h-[220px] flex-1 overflow-y-auto px-4 py-3"
      >
        {items.length === 0 ? (
          <p className="py-6 text-[13px] leading-relaxed text-faint">
            Nothing has been said yet. Silence is a decision here — the speak predictor is in the
            panel, and it will show you why.
          </p>
        ) : (
          <ol className="flex flex-col gap-3">
            {items.map((item) => (
              <Line key={item.seq} item={item} />
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function Line({ item }: { item: FeedItem }) {
  const cut = item.name === "preempted";
  const accent = item.voice === "analyst" ? "border-analyst/45" : "border-caller/45";
  const voiceColour = item.voice === "analyst" ? "text-analyst" : "text-caller";
  const spoken = item.name === "spoken" ? item.text : item.spoken;
  const unsaid = item.text.startsWith(spoken) ? item.text.slice(spoken.length) : "";

  return (
    <li className={`border-l-2 pl-3 ${accent} ${cut ? "opacity-90" : ""}`}>
      <div className="flex items-baseline gap-2">
        <span className={`font-mono text-[10px] uppercase tracking-[0.2em] ${voiceColour}`}>
          {item.voice}
        </span>
        <span className="tnum font-mono text-[10px] text-faint">{videoClock(item.videoTs)}</span>
        {item.event !== "none" ? (
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">
            {item.event.replace(/_/g, " ")}
          </span>
        ) : null}
        {cut ? (
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-reject">cut off</span>
        ) : null}
      </div>
      <p className="mt-1 text-[15px] leading-snug text-text">
        {spoken}
        {unsaid ? <span className="text-faint line-through">{unsaid}</span> : null}
      </p>
    </li>
  );
}
