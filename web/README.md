# web — the watch page

The delayed broadcast, the commentary as it goes to air, and the decisions
behind every line the system did and did not say.

## Running it

```bash
npm install
npm run dev          # http://localhost:3000
```

It expects the Python runtime on `http://127.0.0.1:8000`. From the repo root:

```bash
uv run python -m commentary run --serve --seconds 300
```

That runs the simulated match against the oracle backend, so it needs no API
key and no broadcast on screen. Point somewhere else with
`COMMENTARY_API_ORIGIN=http://host:port npm run dev`.

**With no backend at all:** open <http://localhost:3000/?mock=1>, or use the
`LIVE / FIXTURE` toggle in the header (shown in dev, and in any build when
`?mock=1` is present). That replays `mock/events.ts` — one realistic minute
covering a build-up, a shot, a save whose line the fact gate rejects, a
deliberate silence, and a goal that cuts the analyst off mid-sentence. It runs
through the same parser as the live stream, so it is also the layout fixture.

If the runtime is not up, the status strip says so and keeps retrying with
backoff. It never shows a calm empty page that looks like a quiet match.

## What the agent panel is showing

Audio only carries what the system said. Almost everything interesting about a
commentary system is in what it *didn't* say, and why — so the right-hand
column puts that on screen next to the picture. **Fact gate** lists the lines
that were blocked before they reached a voice, struck through, each with its
reason tags: a name that is on neither team sheet, a scoreline that disagrees
with the board. **Preempted** shows lines the director killed — the part of a
sentence that escaped before a goal took the microphone, with the rest struck
through, plus beats that aged out unspoken. **Speak predictor** shows which
triggers fired and how urgent they were, so a stretch of silence reads as a
decision rather than a hang. **Caller form** shows the structured object behind
the sentence — scene, event, confidence, the names it could actually read —
because a confident wrong call and a hedged right one look identical once
they are spoken. **Board reader** and **ledger** show where the score came from
and what the match has cost, with lag from the beat's live edge.

## Layout

```
app/
  layout.tsx           fonts, metadata
  page.tsx             renders <Watch />
  api/stream/route.ts  SSE passthrough, uncompressed  ┐ see lib/proxy.ts
  api/video/route.ts   MJPEG passthrough              ┘
components/            Watch, VideoStage, Scoreboard, Feed, AgentPanel, ui
lib/
  events.ts            the SSE contract as a discriminated union + parser
  store.ts             bounded state: transcript, rejections, cuts, counts
  useEventStream.ts    one EventSource for the page, batched, backoff reconnect
mock/events.ts         the fixture
```

`useEventStream` is the only thing that opens a stream; every panel reads the
store it returns. Messages are folded in on a 100ms tick rather than per
message, and the arrays are capped (400 transcript lines, 60 per panel) — a 90
minute match emits more than a tab should hold.

`?mock=1` is read on the server, in `app/page.tsx`, not after hydration: the
HTML that comes back is already in fixture mode, so the browser never requests
the MJPEG from a runtime that is not running. In fixture mode the page opens no
connection of any kind.

`allowedDevOrigins` in `next.config.ts` is load-bearing, not decoration. Next
serves its dev bundle only to `localhost` by default, so opening the dev server
on `127.0.0.1` blocks every chunk and the page arrives as server HTML that
never hydrates — a dead toggle and a frozen status strip, with nothing in the
browser console to say why. The only sign is a warning in the `next dev`
output. Both loopback spellings are allowed there now; add any other host you
serve from.

`/api/state` and `/healthz` are proxied by a `rewrites()` entry in
`next.config.ts`. The two streaming endpoints are route handlers instead: a
rewritten response gets gzipped for the browser, and a gzipped
`text/event-stream` buffers in the encoder and never arrives. `curl` saw
events, the page saw an open connection and silence.

## Checks

```bash
npx tsc --noEmit
npx eslint .
npm run build
```
