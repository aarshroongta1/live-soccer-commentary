"use client";

import { useEffect, useState } from "react";

import type { NoticeEvent } from "@/lib/events";
import { secondsAgo } from "@/lib/format";
import type { Connection, StreamSource } from "@/lib/useEventStream";

/**
 * Honest about the thing most dashboards lie about: whether anything is
 * actually arriving. A page with no stream says so, and says where the stream
 * was meant to come from, instead of sitting there empty and calm.
 */
export function StatusStrip({
  connection,
  attempts,
  source,
  lastEventAt,
  notice,
}: {
  connection: Connection;
  attempts: number;
  source: StreamSource;
  lastEventAt: number | null;
  notice: NoticeEvent | null;
}) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const { dot, text } = describe(connection, attempts, source);
  const error = notice?.level === "error";

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
      <span className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        <span className="font-mono text-[11px] tracking-wide text-dim">{text}</span>
      </span>

      {lastEventAt !== null ? (
        <span className="font-mono text-[11px] tracking-wide text-faint">
          last event {secondsAgo(lastEventAt, now)}
        </span>
      ) : null}

      {notice?.text ? (
        <span
          className={`min-w-0 truncate font-mono text-[11px] tracking-wide ${
            error ? "text-reject" : "text-faint"
          }`}
          title={notice.text}
        >
          {notice.text}
        </span>
      ) : null}
    </div>
  );
}

function describe(
  connection: Connection,
  attempts: number,
  source: StreamSource,
): { dot: string; text: string } {
  if (source === "fixture") {
    return { dot: "bg-analyst", text: "replaying fixture · no backend" };
  }
  if (connection === "open") {
    return { dot: "bg-caller live-dot", text: "stream connected" };
  }
  // One failure is enough to say the true thing. "Opening" is only honest
  // before the first attempt has come back.
  if (attempts > 0) {
    return {
      dot: "bg-reject",
      text: `no runtime on :8000 — start it and this reconnects (attempt ${attempts})`,
    };
  }
  return { dot: "bg-dim", text: "opening stream on :8000" };
}
