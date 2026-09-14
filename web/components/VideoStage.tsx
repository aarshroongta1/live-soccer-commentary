"use client";

import { useEffect, useState } from "react";

/**
 * The broadcast, as the commentator sees it.
 *
 * The stream is taken from the delay buffer behind the narration cursor, not
 * at the live edge, so the picture here is the picture the caller is
 * describing — which is the only reason an eight second delay is invisible
 * rather than infuriating. It is a plain `<img>` because
 * `multipart/x-mixed-replace` is decoded by the browser itself; a video
 * element would be wrong twice over.
 *
 * *Behind* the cursor, because a line is written about the cursor and arrives
 * a model round trip later. The runtime holds the picture back by that round
 * trip and says how far in `/api/state`; the corner label prints the figure so
 * the delay on screen is a number somebody can check rather than a claim.
 *
 * The element is mounted only once the event stream is open. A runtime that
 * is not running cannot serve video either, and an `<img>` pointed at a dead
 * endpoint leaves a broken-image glyph sitting in the corner of the picture
 * for the whole session.
 */
export function VideoStage({ live, connected }: { live: boolean; connected: boolean }) {
  const [failed, setFailed] = useState(false);
  const offsetS = usePresentOffset(live && connected);
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
      aria-label="Delayed broadcast, held behind the narration cursor"
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
          {!live
            ? "fixture"
            : offsetS === null
              ? "delayed · narration cursor"
              : `delayed · narration cursor −${offsetS} s`}
        </span>
      </div>
    </div>
  );
}

/**
 * How far behind the cursor the picture is being held, read once from the
 * snapshot endpoint.
 *
 * It is a fixed setting for the life of a run, so one fetch is the whole of
 * it. Null while unknown, and null forever in fixture mode: `?mock=1` must
 * leave the network alone, and there is no runtime behind it to ask.
 */
function usePresentOffset(enabled: boolean): string | null {
  const [offset, setOffset] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled || offset !== null) return;
    const abort = new AbortController();
    fetch("/api/state", { signal: abort.signal, cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((body: unknown) => {
        const seconds = presentOffsetOf(body);
        // A runtime serving the cursor exactly has nothing to add to the
        // label, and "−0 s" reads as a bug rather than as a setting.
        if (seconds !== null && seconds >= 0.05) setOffset(String(Number(seconds.toFixed(1))));
      })
      .catch(() => undefined);
    return () => abort.abort();
  }, [enabled, offset]);

  return offset;
}

function presentOffsetOf(body: unknown): number | null {
  if (typeof body !== "object" || body === null) return null;
  const status = (body as { status?: unknown }).status;
  if (typeof status !== "object" || status === null) return null;
  const seconds = (status as { present_offset_s?: unknown }).present_offset_s;
  return typeof seconds === "number" && Number.isFinite(seconds) ? seconds : null;
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
