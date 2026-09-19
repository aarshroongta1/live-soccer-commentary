# Screen commentary

The `/live` page generates caller and analyst text while a local video or shared
screen plays. It uses GPT-5.6 Terra for both observation and writing. Speech is
not synthesized.

## Run

Follow the [setup steps](../README.md#run-locally), including `OPENAI_API_KEY`,
then open http://127.0.0.1:3000/live in a desktop browser.

Choose a local video or select **Share match screen** and pick the match tab or
window. Screen sharing may require browser and OS permission. Select
**Generic match**, confirm the paid-call notice, and press **Start commentary**.
Selecting a source alone makes no model calls. **Stop** ends capture and the worker.

Generic mode needs no extra files and cannot resolve roster names. The
**Argentina–France** preset is specific to the 2022 final and requires two local
files that are not bundled with the repository:

- `clips/pack-argfra-2022.json`: team and player identities.
- `runs/demo/argfra-all-openai-v1-plan.json.run/research.json`: saved research.

## Processing

The browser captures one JPEG per second and retains at most four unsent frames.
Batches are sent at least four seconds apart, with one request in flight.
Old unsent frames are dropped when processing falls behind.

A Next.js API route manages a persistent Python worker over JSONL. The worker
runs the observe → write LangGraph and retains recent evidence and commentary.
There is no separate backend server to start.

## Limits

Sessions run locally, one at a time, and stop after 60 seconds, 12 batches, or
an estimated $0.35 in tokens. Cost is checked after each batch, so the final
request can exceed that estimate. Timeouts clean up abandoned workers.

Sampled frames go to OpenAI. Microphone and system audio are not captured, and
normal sessions are not saved to disk. Commentary trails the action by several
seconds and can misidentify players or events. The displayed request latency
measures processing time, not the full delay from an action to its commentary.

For a terminal experiment with saved latency and usage logs, use
`scripts/try_streaming.py --help`. That command is a dry run unless `--spend` is
supplied.
