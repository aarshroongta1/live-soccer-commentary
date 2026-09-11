"use client";

import { useEffect, useState } from "react";

/**
 * The broadcast, as the commentator sees it.
 *
 * The stream is taken from the delay buffer at the narration cursor, not the
 * live edge, so the picture here is the picture the caller is describing —
 * which is the only reason an eight second delay is invisible rather than
 * infuriating. It is a plain `<img>` because `multipart/x-mixed-replace` is
 * decoded by the browser itself; a video element would be wrong twice over.
 *
 * The element is mounted only once the event stream is open. A runtime that
 * is not running cannot serve video either, and an `<img>` pointed at a dead
 * endpoint leaves a broken-image glyph sitting in the corner of the picture
 * for the whole session.
 */
export function VideoStage({ live, connected }: { live: boolean; connected: boolean }) {
  const [failed, setFailed] = useState(false);
  // A new mount token forces the browser to re-request the MJPEG after a
  // failure rather than serving the dead response back from cache.
  const [token, setToken] = useState(0);

  useEffect(() => {
    if (!failed) return;
    const timer = window.setTimeout(() => {
      setFailed(false);
      setToken((value) => value + 1);
    }, 4000);
    return () => window.clearTimeout(timer);
  }, [failed]);

  const showPicture = live && connected && !failed;

  return (
    <div
      role="img"
      aria-label="Delayed broadcast at the narration cursor"
      className="relative aspect-video w-full shrink-0 overflow-hidden border border-line bg-black lg:max-h-[52vh]"
    >
      {showPicture ? (
        // The alt text is empty on purpose: a broken MJPEG would otherwise
        // print a sentence across the picture every time the runtime blinks.
        // eslint-disable-next-line @next/next/no-img-element -- MJPEG, decoded by the browser
        <img
          key={token}
          src="/api/video"
          alt=""
          className="h-full w-full object-contain"
          onError={() => setFailed(true)}
        />
      ) : (
        <Placeholder live={live} connected={connected} />
      )}

      <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-sm bg-black/55 px-2 py-1 backdrop-blur-sm">
        <span className={`h-1.5 w-1.5 rounded-full ${showPicture ? "bg-caller live-dot" : "bg-faint"}`} />
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-dim">
          {live ? "delayed · narration cursor" : "fixture"}
        </span>
      </div>
    </div>
  );
}

function Placeholder({ live, connected }: { live: boolean; connected: boolean }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-[radial-gradient(circle_at_50%_40%,#101619,#05070800)] text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-dim">no picture</p>
      <p className="max-w-sm px-6 text-[13px] leading-relaxed text-faint">
        {!live
          ? "Replaying the fixture. The events beside this are real in shape but recorded; there is no broadcast behind them."
          : connected
            ? "The runtime is answering but the delayed video stream is not. The picture will come back on its own."
            : "No runtime on :8000, so there is no delay buffer to show. Start it and the picture appears here."}
      </p>
    </div>
  );
}
