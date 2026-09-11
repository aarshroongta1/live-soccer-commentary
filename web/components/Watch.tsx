"use client";

import { useState } from "react";

import { AgentPanel } from "@/components/AgentPanel";
import { Feed } from "@/components/Feed";
import { Scoreboard } from "@/components/Scoreboard";
import { StatusStrip } from "@/components/StatusStrip";
import { VideoStage } from "@/components/VideoStage";
import { useEventStream, type StreamSource } from "@/lib/useEventStream";

/**
 * The page. One stream, one store, three readers of it: the picture, the
 * transcript, and the panel that shows the decisions behind the transcript.
 */
export function Watch({ initialSource }: { initialSource: StreamSource }) {
  // The server already decided from `?mock=1`; the toggle only overrides it
  // from here on, so the first render and the markup it hydrates agree and no
  // live request is made in fixture mode.
  const [source, setSource] = useState<StreamSource>(initialSource);

  const { store, connection, attempts } = useEventStream(source);
  const showToggle = initialSource === "fixture" || process.env.NODE_ENV !== "production";

  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-line px-4 py-3 sm:px-6">
        <h1 className="font-mono text-[11px] uppercase tracking-[0.28em] text-text">
          Live Soccer Commentary
        </h1>
        <StatusStrip
          connection={connection}
          attempts={attempts}
          source={source}
          lastEventAt={store.lastEventAt}
          notice={store.notice}
        />
        {showToggle ? <SourceToggle source={source} onChange={setSource} /> : null}
      </header>

      <main className="grid min-h-0 flex-1 gap-3 p-3 sm:p-4 lg:grid-cols-[minmax(0,1fr)_400px] xl:grid-cols-[minmax(0,1fr)_440px]">
        <div className="flex min-h-0 flex-col gap-3">
          <VideoStage live={source === "live"} connected={connection === "open"} />
          <Scoreboard state={store.state} />
          <Feed items={store.feed} />
        </div>

        <aside className="min-h-0 lg:overflow-y-auto lg:pr-1">
          <AgentPanel store={store} />
        </aside>
      </main>
    </div>
  );
}

function SourceToggle({
  source,
  onChange,
}: {
  source: StreamSource;
  onChange: (next: StreamSource) => void;
}) {
  return (
    <div className="flex items-center overflow-hidden rounded-sm border border-line">
      {(["live", "fixture"] as const).map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onChange(option)}
          aria-pressed={source === option}
          className={`px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.16em] transition-colors ${
            source === option ? "bg-raise text-text" : "text-faint hover:text-dim"
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}
