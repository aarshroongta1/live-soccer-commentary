# Live search → fact → commentary

Public replay: **https://soccer-commentary-demo.vercel.app/demo/keenable**.
This route always uses saved search events and generated audio, with selectable
workflow text and no live API calls. It needs no login or API keys. The normal
localhost route below still runs the live pipeline.

Open **http://localhost:3000/demo/replay** and play the video normally.
Only the match footage is video. The event, query, search responses, writer input
and generated line are selectable HTML text that updates as events arrive.
At the reviewed goal cue, 4 seconds into the clip, the browser starts a real run:

1. Keenable and Exa receive the same query in concurrent requests.
2. Each response appears when it actually arrives, with its ranked titles and
   verbatim excerpts. The UI displays the beginning of each excerpt with an
   ellipsis; all returned text is available under the result expander and raw trace.
3. The researcher selects a source passage from the first successful response.
4. That exact passage and the reviewed goal event go to the commentary writer.
5. The exact generated football line is synthesized and played over the footage.

The screen reads left to right: footage and reviewed event → shared query and
provider branches → commentary input and output. Blue connects the event's
player/team names to the literal query. Green connects the selected result to
the exact writer input, then highlights shared phrases (including possessive club names) in the evidence
and generated line. The labels and highlights show the handoffs without arrows
or transfer animations. These marks do not change the text.

The selected provider and result number remain visible at the handoff. Request,
response, source passage, writer input, generated line, model usage and voice
credits are retained under `runs/outreach/live/<run_id>/trace.json`.

The video keeps its original match audio. When the AI analyst speaks, that audio
is lowered, then restored. There is no provider-promotion voiceover and no
editorial rewrite of the generated analyst line. The goal/scorer is explicitly
human-reviewed for this demonstration; it is not a new vision inference claim.

## Actual outputs and timing

Earlier versions displayed the same two manually written fact summaries beneath
both provider names. Those summaries were supported by saved results, but were
not literal provider outputs. They have been removed from the UI. The current
interface uses actual response titles/excerpts and the actual selected passage.

The winner is not forced. Two retained development runs illustrate the variation:

| Actual run | Keenable | Exa | Context used |
| --- | ---: | ---: | --- |
| `20261115T030006-8b943a` | 536.7 ms | 2,470.0 ms | Keenable |
| `20261115T030522-657edd` (browser run) | 772.3 ms | 296.1 ms | Exa |

Both used the same query:

> Dan Ballard Sunderland Arsenal 8 November 2025 background former club Arsenal defensive record before match

Keenable used its default pro search with snippets; Exa used auto with highlights
and its recommended objective field. Both requested five results. These are
single-request client timings, not a benchmark or a universal speed claim. The
browser run is retained even though Exa won. No extra calls were made to search
for another winner.

The current saved-search replay uses the original Keenable-first responses and
arrival offsets with a new selector, writer and voice run (`20261115T044137-e472ae`).
The selector now favors passages that explicitly connect the scorer to the opponent.
The writer explains why that background matters to this goal, naming the club
rather than relying on an academy nickname or reciting career dates. It avoids
profile-style introductions such as “formerly of” and reacts to the goal directly.

The selector chose this exact passage from Keenable result #3 (The Guardian):

> The goal from Ballard, a former Arsenal academy player, ended a run of eight successive clean sheets in all competitions for Arsenal.

The model generated, and the voice speaks, this exact line:

> Dan Ballard has come back to haunt Arsenal there — a former Arsenal academy player ending their eight-game clean-sheet run.

This text was not supplied as a target answer or rewritten after generation.
The original two live runs remain in the audit folders, including the Exa-first
counterexample. The UI labels the revised version “saved searches”; the search
timings are recorded, while the commentary and voice are newly generated.

This is live research about historical footage. Search results are from today's
indexes; source passages used by the researcher must have publication dates no
later than 8 November 2025. Day-level date metadata is not a guarantee of
historical intraday availability or page immutability.

## Live mode and costs

The localhost-only `POST /api/search-commentary` streams actual worker events as
NDJSON. No keys enter the browser. The worker uses the existing search adapters,
OpenAI backend, usage reservations and ElevenLabs voice configuration.

A live page starts at most one run, when playback reaches the goal. Seeking or
replaying on that page reuses its results. Reloading creates a new opportunity
to run. There is a three-run limit per server session and no automatic retries.

Each run allows two searches, two `gpt-5.4-mini` calls with a combined **$0.05
model cap**, and one `eleven_flash_v2_5` call capped at **240 characters**. The
selector returns only a passage ID; code copies the passage and URL from the
provider result. The writer receives that passage and the human-reviewed event.
Provider/workflow narration is rejected before synthesis.

The first new run used $0.00381075 in model tokens, $0.007 reported Exa search
cost, one free Keenable credit, and 64 voice credits. The browser run used
$0.00326175 in model tokens, $0.007 Exa search cost, one free Keenable credit,
and 70 voice credits. TTS did not return an account-specific dollar invoice.
The revised saved-search evaluation made zero new search API calls. Its two
`gpt-5.4-mini` calls cost $0.00097065 (including cached input) and the voice used 62 character credits.
These calls and earlier development work remain within the authorized $1 cap.

Regression tests use mocked providers and ScriptedBackend, never paid calls.
They verify that the second search can report while the first result is being
processed, that completion order is not forced, that invalid passages cannot
reach the writer, and that promotional narration cannot reach voice synthesis.

## Replay the Keenable-first run without paying again

Open **http://localhost:3000/demo/replay?mode=replay**. This replays the first
run's search timing with the revised generated commentary and speech; playback
does not make live search/model calls. The header labels this as saved searches.

- Source footage: [Arsenal's official highlights](https://www.youtube.com/watch?v=SXHoDl2twPg), source seconds 1.5–21.5 (the goal precedes the four-second reviewed cue).
- Browser player: `web/public/soccer-commentary/ballard-live-replay.mp4`.
- Recorded events: `web/public/soccer-commentary/live-search-run.json`.

The original broadcast media and generated audio/video stay gitignored.
The published trace includes full provider responses and exact event records.

To regenerate the match audio and replay trace from the retained actual run:

```bash
.venv/bin/python scripts/prepare_live_search_replay.py \
  --trace runs/outreach/live/20261115T044137-e472ae/trace.json
```

The replay mux places speech at its actual ready offset (9.878 seconds into the
clip for this revision), preserves exact words, and lowers original audio while it
plays. No speed change was needed: speech ends at 16.519 seconds. All workflow text
is rendered by the page from the trace, separate from the match video.

To capture the whole workflow manually, record the browser window and use the
video's normal play control. A recorder with system-audio support is needed to
capture both audio tracks. The player's fullscreen control shows footage alone.

## CLI

```bash
# Dry run: no requests.
.venv/bin/python scripts/live_search_commentary.py

# Regenerate commentary from recorded searches (paid models/voice; zero search calls).
.venv/bin/python scripts/live_search_commentary.py --spend \
  --replay-searches runs/outreach/live/20261115T030006-8b943a/trace.json

# A new paid live run, with the caps above; emits JSON lines.
.venv/bin/python scripts/live_search_commentary.py --spend
```

Set `KEENABLE_API_KEY`, `EXA_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY` and
`ELEVENLABS_ANALYST_VOICE` in the gitignored root `.env`.
