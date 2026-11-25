# Graphon: search the commentary review history

This demo uses the existing repo's review reports as a private, searchable archive.
It helps a developer find why a call failed and what was already tried. It does
not change observation, commentary, memory prompts, or the public website.

## Directions considered

| Direction | Existing material reused | Value and tradeoff |
| --- | --- | --- |
| Review archive (chosen; small) | False-goal investigation, memory transfer results, writer reviews | Find related failures in ordinary language, then read the exact source version and passage. Smallest useful test without new model generation. |
| Analyst research archive (later) | `SearchRun`, cited research briefs, `prepare_research` | Reuse previously collected background before fresh web search. Needs date/fixture filtering and quote validation; existing Keenable/Exa retrieval already works. |
| Footage search (exploratory) | Existing short clips and their source windows | Graphon's media search can return timestamped matches. Needs a direct test on football footage; retrieval would not independently establish a goal or player identity. |

For eight local reports, `rg` and reading files remain simpler. Graphon's potential
advantage is a shared archive accessible to agents outside this checkout, semantic
retrieval, and versioned source spans. This smoke test does not establish superiority
over local search, recognition improvement, or learned generalization.

The [Graphon search documentation](https://docs.graphon.ai/api/search/search-objects)
describes hybrid/keyword retrieval and versioned spans; text is read separately
with `read_spans`. [Media search](https://docs.graphon.ai/use-cases/media-search)
describes timestamped retrieval. These are documented capabilities, not measured
football recognition results.

## Corpus and boundaries

Workspace: `commentart` (`wspxw6nmll3ic5d`). Bucket:
`commentary-review-demo` (`bktrkijHHuLD9`), private, indexing enabled, GCP, US
multi-region. Other providers and locations are available. Open the workspace in
the [Graphon console](https://app.graphon.ai).

Eight reports, 44,258 bytes, copied without rewriting. `SOURCES` in
`scripts/graphon_review_demo.py` is the complete allowlist. Plain-text copies use
the original repo path plus `.txt`. The local manifest records original paths,
byte counts, SHA-256 hashes, and the three questions fixed before querying.
Only the `corpus/` folder is uploaded; the manifest and expected-file checks are
not in Graphon's search inputs.

These are retrospective engineering reports. They can describe known outcomes
and reviewed failures. They are for offline review only and are never provided
to the runtime caller, observer, or recognition evaluation as evidence. No footage,
secrets, raw provider responses, or reference-label JSON is uploaded.

## Try it

From an authenticated Graphon MCP session:

1. `get_bucket_index_status`, workspace `wspxw6nmll3ic5d`, bucket `bktrkijHHuLD9`.
   Inspect the counts: during this upload the summary said `ready` with only
   six of eight objects ready. Wait for all eight and zero unfinished objects
   for a complete-corpus comparison, or explicitly label a partial-index run.
2. `search_objects`, same workspace, target `bktrkijHHuLD9`, mode `hybrid`,
   resultType `passage`, limit `3`, query:
   “Did saved lessons improve recognition on a different later penalty?”
3. Pass returned spans unchanged to `read_spans`, with `include: ["text"]`.
4. Read the report's qualifications alongside its result. A retrieved lesson is
   not independent evidence about the current match.

Or use the official CLI (each search is a new service request):

```sh
graphon --workspace wspxw6nmll3ic5d search commentary-review-demo \
  'Did saved lessons improve recognition on a different later penalty?' \
  --mode hybrid --result-type passage --limit 3
```

No app API key was created. MCP and CLI use their existing OAuth logins.

## Reproduce the offline preparation

```sh
# Choose a new output directory; preparation refuses to overwrite a snapshot.
.venv/bin/python scripts/graphon_review_demo.py \
  --archive runs/graphon-review/new-snapshot

# Review this inventory before uploading to a chosen bucket.
graphon --workspace wspxw6nmll3ic5d objects sync commentary-review-demo \
  --dir runs/graphon-review/new-snapshot/corpus --dry-run
```

The original upload was followed by another dry run: **would upload 0, skipped 8,
failed 0**. The source files remain untouched. No deletion is part of this demo.

Saved MCP search/read responses can be replayed without service calls:

```sh
.venv/bin/python scripts/graphon_review_demo.py \
  --archive runs/graphon-review/demo-v1 \
  --capture examples/graphon-review/capture.json
```

The replay checks source hashes, allowed keys, unchanged returned spans, and read
completeness. Each passage displays whether it exactly equals the archived
Unicode character slice. Tests use local fixtures and make no paid calls.

## Results

The [captured responses](../examples/graphon-review/capture.json) retain all three
queries and all nine read passages, including irrelevant lower-ranked results.
Each passage exactly matches its original Unicode character slice. No search
rerolls, generated answers, commentary model calls, TTS, or deep research ran.

| Fixed question | Observed retrieval | Service search time |
| --- | --- | --- |
| Why can a goal call have valid citations and still be false? | Text evaluation report at rank 1; the predeclared false-goal investigation at ranks 2 and 3. Both explain that invented observations and self-reported cue tags can pass structural checks. | 1,426 ms |
| Did saved lessons improve recognition on a different later penalty? | Parent lesson report at rank 1 states that later-incident transfer did not establish improvement. The predeclared detailed transfer report was not searchable yet. | 1,410 ms |
| Why was the optional reviewer turned off for penalty commentary? | Predeclared portfolio report at rank 1 explains the reviewer rejected a valid short goal call. | 1,386 ms |

**Partial indexing:** after about eight minutes, six files were ready and two
remained `indexing/processing`: `examples/lesson-memory/transfer/README.md.txt`
and `examples/keenable-outreach/README.md.txt`. There were zero failed or unsupported
files. The searches above ran once against that state; the record does not
silently substitute a later, fully indexed run. Graphon returned no warning about
the partial index on the successful searches. Two of three predeclared expected
source files appeared in the top three passages; the third was still indexing.
That third check is incomplete, not evidence of failed semantic ranking.

These are navigation smoke checks over a deliberately selected corpus, not a
retrieval benchmark or an evaluation of generated answers. Times are single-run
service timings; client search wall times were 1.64–1.91 seconds, and passage
reads took 0.72–1.28 seconds. Search/read responses exposed no dollar charge, so
no exact Graphon cost is claimed. No separately billed commentary models ran.

## Assessment

This is useful as a shared engineering review archive: an agent can locate the
prior investigation without having this checkout or remembering filenames, then
read the original passage and version. It does not warrant moving the live
commentary pipeline to Graphon. For this tiny corpus, local file search remains
a good default. A larger archive of reviewed run summaries would make the shared
retrieval more valuable; fresh uploads must be checked for indexing completeness.

The next useful extension would be one summary per reviewed run linking to its
retained trace, followed by a comparison on questions not chosen from the indexed
reports. Analyst background retrieval needs its own date/fixture controls.
