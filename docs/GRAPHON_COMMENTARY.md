# Graphon in the commentary workflow

The visual demo follows the existing Keenable, Unsiloed, and Auctor demos:
show original material beside the actual output, highlight the handoff, keep
text selectable, and provide the full record underneath.

Canonical page: `web/public/graphon-commentary/`. The same files are mirrored
into `../personal-website/public/graphon-commentary/`; that repo's Vercel rewrites
map `/demo/graphon` and `/demo/graphon/` to the page. No production deployment
is part of preparing the local preview.

## What happens

1. The existing Sunderland–Arsenal clip plays with its retained caller voice.
2. At the reviewed goal, the replay reveals the query:
   `Dan Ballard Arsenal academy background`.
3. Graphon's actual hybrid search returns two saved pre-match research excerpts.
4. A GPT-5.4 Mini selector chooses a verbatim paragraph from the returned
   Ballard interview. Code copies it unchanged into the writer's input.
5. The writer receives only the reviewed goal and that cited paragraph. Its
   exact generated line is synthesized in the existing analyst voice.

Exact selected passage:

> He wasn’t having the easiest of times, having been released by Arsenal before
> suffering a broken ankle, which caused other trials to fall through.

Exact generated commentary:

> Dan Ballard, once released by Arsenal, has now come back to score against
> them for Sunderland.

The original caller words and audio are reused. The new analyst performance
starts at 9.8779s and ends at 14.429011s. All speech plays at natural speed,
without overlapping voices. Original broadcast audio is discarded by the mux.
On-screen words are HTML, not burned into the footage.

## What Graphon contributes

Graphon stores the archived research and finds passages relevant to the player
and opponent. It replaces a fresh web search at that point in the workflow.
It does not observe the goal or identify the player. The scorer/event is a
reviewed input, as in the Keenable demo. The selector and writer use the existing
model/backend/budget patterns; Graphon is responsible for retrieval.

The archive has only two excerpts, so this demonstrates a working handoff,
not an advantage over other retrieval systems. A fuller match-research library
would make retrieval more useful. No claim of improved recognition, autonomous
learning, or live spoken latency is made.

## Actual retrieval and provenance

Workspace: `commentart` (`wspxw6nmll3ic5d`). Private bucket:
`commentary-research-demo` (`bktqAwNd16qE0`), indexing enabled, GCP US multi-region.
Both files finished indexing before the query, with zero failed/unsupported files.

| Rank | Object | Original publication metadata |
| --- | --- | --- |
| 1 | `arsenal-match-preview.txt` | 8 November 2025, 09:00 UTC |
| 2 | `ballard-interview.txt` | 29 October 2025, 06:00 UTC |

The source excerpts already existed in the Keenable/Exa experiment's saved
Exa response, `runs/outreach/live/20261115T044137-e472ae/trace.json`. Their
original URLs, titles, dates, excerpts and snapshot hashes are retained in
`examples/graphon-commentary/sources.json`. They were not newly authored to
contain the desired line. Publication metadata puts both before the scheduled
kick-off; it is not independent proof of historical page immutability.

Graphon returned the preview at rank 1 and the interview at rank 2. The model
selected the latter for the player's history; the rank order was not changed.
The query ran once. `read_spans` received its exact versioned character spans.
Both returned texts exactly match their original Unicode character slices.

Search: **919 ms client wall time**, 578 ms reported service total.
Passage read: **595 ms client wall time**. These are individual captured
requests, not a benchmark. The [captured retrieval](../examples/graphon-commentary/retrieval.json),
[selector input](../examples/graphon-commentary/selector-input.json),
[writer input](../examples/graphon-commentary/writer-input.json), and raw model
responses retain the actual handoffs.

## Replay and cost

This is a saved-run replay, matching the public Keenable demo. Loading or playing
it makes no paid calls. Stage reveals are synchronized to prepared clip timing;
they do not reconstruct a measured live end-to-end run. Raw model timings are
not claimed. The two model calls cost **$0.002001** in reported token usage,
under a $0.03/two-call cap, with no unsettled reservations. One ElevenLabs Flash
v2.5 request synthesized 94 characters and returned 47 character credits. Neither
Graphon nor TTS returned an account-specific dollar invoice.

The first media export stopped because the new plan omitted metadata required
by the existing renderer. The paid model and audio responses had already been
saved. Export was fixed and resumed offline; no model/TTS request was repeated
and no generated words were edited. Raw calls remain retained. A saved run can
be exported again with integrity checks and no paid calls:

```sh
.venv/bin/python scripts/prepare_graphon_commentary.py --export
```

The default invocation is a dry run. `--spend` is explicit and refuses to
overwrite a completed paid attempt. No Graphon app key was created; retrieval
used the connected MCP server and local file upload used the authenticated CLI.

## Local preview and verification

```sh
cd ../personal-website
npm run dev -- --host 127.0.0.1 --port 5176
# Open http://127.0.0.1:5176/graphon-commentary/index.html
```

The explicit `index.html` matters for local Vite serving. Production uses the
Vercel rewrite. Vite serves byte ranges for video seeking.

Offline tests verify source versions, original character ranges, date boundaries,
the selected-quote handoff, exact model-output preservation, and speech audit
consistency. Browser checks cover playback, seeking, full-run inspection,
desktop/mobile layout, readable source text, and the absence of runtime errors.

```sh
.venv/bin/python -m pytest tests/test_graphon_commentary.py -q
```

This commentary demo replaces the earlier debugging-archive experiment as the
user-facing Graphon direction. That experiment's files and private bucket are
retained separately; none of its retrospective reports enter this workflow.
