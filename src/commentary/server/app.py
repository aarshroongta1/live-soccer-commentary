"""The window onto a running match.

Two streams leave this process. One is the feed of everything the agents did,
as server-sent events. The other is the video itself, taken from the delay
buffer at the narration cursor rather than at the live edge — the viewer and
the commentator therefore see the same instant, which is the whole reason the
delay is invisible rather than annoying.

Neither stream may slow the match down. The bus drops messages for a slow
subscriber, and the video encoder skips frames rather than queueing them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from commentary.bus import Bus
from commentary.capture.buffer import DelayBuffer
from commentary.llm.base import Usage
from commentary.schemas import MatchState

BOUNDARY = "frame"


class RuntimeHandle(Protocol):
    """What the web layer is allowed to know about a running match."""

    @property
    def bus(self) -> Bus: ...

    @property
    def buffer(self) -> DelayBuffer: ...

    @property
    def state(self) -> MatchState: ...

    @property
    def usage(self) -> Usage: ...

    def status(self) -> dict[str, Any]: ...


async def mjpeg(buffer: DelayBuffer, fps: int = 12, quality: int = 72) -> AsyncIterator[bytes]:
    """The delayed video, as a multipart JPEG stream.

    Deliberately re-encodes from the cursor on a fixed wall-clock tick instead
    of following the buffer: if the browser falls behind, it should see the
    present late, not the past in order.
    """
    from commentary.llm.base import encode_frame

    interval = 1.0 / fps
    last_ts = -1.0
    while True:
        cursor = buffer.cursor_ts
        frame = buffer.nearest(cursor) if cursor is not None else None
        if frame is not None and frame.ts != last_ts:
            last_ts = frame.ts
            jpeg = encode_frame(frame.image, quality=quality, max_width=960)
            yield (
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode() + jpeg + b"\r\n"
        await asyncio.sleep(interval)


def create_app(runtime: RuntimeHandle) -> FastAPI:
    app = FastAPI(title="Live Soccer Commentary", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return {
            "state": runtime.state.model_dump(mode="json"),
            "status": runtime.status(),
            "usage": {
                "input_tokens": runtime.usage.input_tokens,
                "output_tokens": runtime.usage.output_tokens,
                "cost_usd": round(runtime.usage.cost_usd, 4),
            },
        }

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            # An immediate comment so the browser's EventSource opens even if
            # the match is in a quiet spell.
            yield ": connected\n\n"
            async for message in runtime.bus.subscribe():
                yield message.sse()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/video")
    async def video() -> StreamingResponse:
        return StreamingResponse(
            mjpeg(runtime.buffer),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(FALLBACK_PAGE)

    return app


#: Served when the Next.js build is not present, so `uv run commentary-serve`
#: is useful on its own without a node toolchain.
FALLBACK_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Live Soccer Commentary</title>
<style>
 body{margin:0;background:#0b0f0c;color:#e8efe9;font:15px/1.5 ui-sans-serif,system-ui}
 main{max-width:1100px;margin:0 auto;padding:24px;display:grid;gap:16px}
 img{width:100%;border-radius:10px;background:#000}
 #feed{display:flex;flex-direction:column-reverse;gap:8px;max-height:46vh;overflow:auto}
 .line{padding:8px 12px;border-radius:8px;background:#16201a}
 .line.analyst{background:#1b1a26}
 .line.cut{opacity:.55;text-decoration:line-through}
 .who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;opacity:.6}
 h1{font-size:16px;font-weight:600;margin:0}
 #score{font-variant-numeric:tabular-nums;opacity:.85}
</style>
<main>
  <header><h1>Live Soccer Commentary <span id="score"></span></h1></header>
  <img id="video" src="/api/video" alt="delayed broadcast">
  <div id="feed"></div>
</main>
<script>
const feed = document.getElementById('feed');
const score = document.getElementById('score');
const es = new EventSource('/api/stream');
function add(cls, who, text) {
  const el = document.createElement('div');
  el.className = 'line ' + cls;
  el.innerHTML = '<div class="who">' + who + '</div>' + text;
  feed.prepend(el);
  while (feed.children.length > 80) feed.lastChild.remove();
}
es.addEventListener('spoken', e => {
  const d = JSON.parse(e.data);
  const who = d.voice || 'caller';
  add(who, who + ' · ' + (d.ts || 0).toFixed(1) + 's', d.spoken || d.text);
});
es.addEventListener('preempted', e => {
  const d = JSON.parse(e.data);
  if (d.reason === 'cut') add('cut', 'cut off', d.spoken || d.text);
});
es.addEventListener('state', e => {
  const d = JSON.parse(e.data);
  if (d.home) score.textContent = d.home + ' ' + d.home_score + '-' + d.away_score + ' ' + d.away
    + (d.clock ? '  ' + d.clock : '');
});
</script>
"""
