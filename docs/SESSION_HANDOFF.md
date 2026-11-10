---
status: ready-for-demo
branch: feature/replay-agents
timestamp: 2026-11-10
files_modified: []
---

# Commentary project handoff

## Latest: concise engineering deep dive

User asked for slightly deeper “How I built it” content with AI-engineering resume
skills and GitHub README-style flowcharts, without making the main demo verbose.
/demo/engineering is now a ~500-word case study: two rendered Mermaid diagrams,
a skills-to-implementation table, three evaluation-driven decisions and a truthful
resume summary. Skills map to actual multimodal inference/OpenCV, LangGraph agent
orchestration, Pydantic structured outputs and grounding, application-managed tools,
asyncio coordination in the separate replay prototype, evaluation, observability
and cost control. Recorded-vs-replay boundaries and narrow evaluation claims remain.

The prior full interactive workbench moved to /demo/engineering/runs, linked by
“Inspect the saved runs”. The ~49-word recruiter showcase at / and /demo is unchanged.
Mermaid 12.3.0 is a pinned dependency, dynamically imported only by the case-study
component. Sources live in web/data/engineering-diagrams.json, match the two
README Mermaid blocks and docs/ENGINEERING_OVERVIEW.md, and are downloadable as
.mmd files. The public engineering-overview.md download includes the same diagrams.
Case study files: web/app/demo/engineering/{case-study.tsx,mermaid-diagram.tsx,
engineering.css}. No runtime agents, model calls, footage or generated words changed.

Validation: production lint/build passed; both Mermaid diagrams rendered in
light/dark themes with no console errors. Mobile diagrams scroll inside their
frames, with no page overflow. Source toggles work; Markdown/.mmd downloads return
200; navigation to the retained saved-run inspector works. The recruiter view
remains minimal with zero technical sections. README and downloadable Markdown
diagram blocks exactly match the JSON sources. Production npm audit reports zero
vulnerabilities; remaining audit findings are pre-existing development lint
dependencies. Screenshots and browser-qa.json are in runs/engineering-deep-dive.


## Latest correction: recruiter-facing showcase

User clarified that the developer workbench exposed too much text, backend work
and notes for recruiters. The main / and /demo routes are now a simple showcase:
name/navigation, title, one sentence, three clip choices, a large video, brief
AI/light-edit and recorded/silent labels, and a “How I built it” footer link.
No inspector, cost data, technical notes, pipeline, evaluation or comparison
controls are on the main view. The full prior workbench remains at
/demo/engineering. Do not reintroduce its technical content on the main view.

New showcase is demo-experience.tsx + showcase.css. The former UI is preserved in
engineering-experience.tsx; demo-data.ts supplies both server pages. No generated
text, media or experiments changed. Both pages retain light/dark themes. The
separate engineering page returns to /demo via “Back to demo”. Recruiter videos
are re-rendered as showcase-{penalty,arsenal,argentina}.mp4 with an empty title
label, so no backend/editing notes are burned above the footage. Generated
commentary words, source windows and cues are unchanged. Engineering retains
its original annotated media. Each new render has tracked ASS and render JSON.

Validation: web lint/build passed. Browser checks confirmed 49 interface words,
zero technical sections on the main route, clip switching pauses actual playback,
light/dark themes and no mobile overflow, the engineering link opens the retained
inspector, and all three clean media URLs return 200. Render reports confirm
identical commentary/cues/source hashes and no audio. No paid calls.



## Latest: developer-themed presentation, 10 November 2026

The user requested a proper showable, developer-themed demo using their website
as reference. Read the live aarshroongta.com and local personal-website source;
no personal-website files were changed. The commentary demo now uses the same
Space Grotesk, blue accents, restrained rules, left navigation rail and light/dark
palette. Default light; theme toggle stays within this page, leaving /live alone.

A footage-first workspace has Transcript / Evidence / Run inspector tabs.
Transcript lines and evidence timestamps seek actual footage; arrow keys and
Home/End navigate the inspector tabs. Run metadata uses saved costs, models and
rejected-turn counts. It explicitly labels recorded preparation and makes no
model calls. Clip selection resets the inspector and pauses the old player.
England retains compact before/after controls; detailed iteration results are
collapsed initially. Light-edit and preparation notes remain downloadable.
Pipeline is reduced to four explainable stages, followed by evaluation findings
and the separate expandable causal replay explorer. Generated words/media and
all experiment records are unchanged. No paid calls or deployment.

Design brief: docs/DEMO_DESIGN.md. Presentation code: web/app/demo/
{demo-experience.tsx,demo.css,architecture.tsx}. Scoped theme tokens replace the
old layered white/green CSS. The walkthrough's underlying story is unchanged.

Validation: production web lint/build passed. gstack checks at 1440, 1280, 768,
375 and 320px found no horizontal overflow. Verified both themes, keyboard
Arrow/Home navigation, transcript and evidence seeks, live source-time display,
per-clip model/cost metadata, playing-video pause on clip switch, England
before/after trace changes, all trace/edit/document downloads (HTTP 200), and
media-error fallback retaining transcript/evidence. No console errors. Screenshots
and browser-qa.json are in runs/demo-workbench. The production server is running
at http://127.0.0.1:3000/demo. Core Python code and all generated media are unchanged;
no paid or Python regression runs were needed for this presentation-only change.


## Current priority: showable portfolio demo — completed locally

The user wants a demo they can show, explain and put on a resume. Light editing is
allowed; voice and open-ended research remain excluded. The focused `/` and
`/demo` pages now feature generated commentary for three clips:

- England–France, 20s: exact unedited model words on saved perception/research.
  The before/after switch retains the optional reviewer failure and restored
  caller goal turn. Same saved evidence, 17 vs 9 calls, 33 words each; later run
  starts at 3s instead of 1s. This is one development pair, not a suite-wide gain.
- Arsenal–Norwich, 18s: fresh vision and independent speaker turns. All three
  retained lines are verbatim. Trim the original 28s to one incident and move the
  analyst cue from 17.000s to 11.508s, after evidence availability. This is a carry
  and finish, not the earlier manually described Wilshere passing sequence.
- Argentina–France, 20s: fresh vision/speakers, select the last 20s of the original
  28s, omitting an earlier analyst line outside the crop. Remove “’s number 9” from
  one caller line. Other retained words and source-relative cues are unchanged.

No football action, name, statistic or explanation was added. Source cuts and
retiming are editorial. Full-context vision saw the original 28s for both new
clips. These are selected examples, not untouched end-to-end quality evidence.
The older heavily rewritten showcases/media/plans remain archived, not featured.
All three clips are silent and captioned; no TTS or measured live speech latency.

Fresh experiment: four attempts capped at $0.20 each, $0.80 maximum total; reported
cost **$0.27827**, all reservations settled, 20 model calls, no retries/web research.
Initial observers and all speakers use gpt-5.4-mini; follow-up observers use
 gpt-5.4-2026-03-05. Select Arsenal attempt two and Argentina attempt one. Stronger
vision was not consistently better: the Argentina follow-up returned five cues
where the schema allows four and failed validation. All attempts remain preserved.
No more model calls are planned. Inputs contain only footage and team/kit context,
empty research briefs, no earlier scripts or evaluator labels. Source review was
by the coding assistant against frames, not an independent human rating.

Artifacts: examples/portfolio-demo/release contains raw plans, featured plans,
exact edit records, all four traces, inputs and selection/spend report.
web/data/featured-demos.json supplies new examples; portfolio-demo.json retains
England. Each clip switches to its own trace, evidence and turn list. New public
media: generated-arsenal.mp4/.jpg and generated-argentina.mp4/.jpg, with tracked
ASS/render/edit/trace JSON. Media remains git-ignored. A fresh clone still shows
transcripts without media. Renderer now preserves a plan's words_edited and
source_time_edits metadata instead of always claiming no word edits.

**docs/DEMO_WALKTHROUGH.md** is the recommended entry point: a 30-second explanation,
two-minute presentation, four code sections to understand, interview questions and
one resume bullet. No trained-model, benchmark-accuracy or proven multi-agent
quality claim. docs/PORTFOLIO_CASE_STUDY.md and docs/PORTFOLIO.md are updated.

Shareable local package: runs/portfolio-release/commentary-demo.zip (38.2 MiB),
three featured clips, England before comparison, walkthrough and edit/cost records.
Individual copies are in runs/portfolio-release/share. Browsing/playing is free.
This is ready to show locally or share as MP4s. No push, deployment or separate
personal-website update has been performed; the existing public page is older.

Validation: **384 offline tests passed**, Ruff and mypy passed, web lint/build
passed. New provenance checks verify preserved words, the disclosed identity
removal, available evidence after retiming, readable captions, and costs including
failed/unselected attempts. ffprobe confirms new video durations 18s/20s, no audio.
gstack browser checks at 1280px/375px: three labelled choices, correct transcripts,
11.508s seek, old-player pause while actually playing, correct per-clip traces and
edits, England before/after, download HTTP 200s, no overflow or console errors.
QA screenshots are in runs/portfolio-release. Production preview is at
http://127.0.0.1:3000/demo; verify its process before reuse. Stop next-server before
rebuilding .next. Generate public media before the static build.

Further work should follow actual user feedback or an explicit release request.
Do not reopen the false-goal rabbit hole. Paid experiments up to $1 are authorized;
ask above $1 and do not split a larger experiment to bypass that limit.

## Latest: false-goal investigation, 9 November 2026

The user asked whether the reason behind false goals can be diagnosed. The
[focused investigation](FALSE_GOAL_INVESTIGATION.md) and
[raw diagnostic record](../examples/evaluation-results/false-goal-investigation-20261109.json)
now separate confirmed propagation mechanisms from unproven visual causes.
Source JPEGs matched saved hashes; source is 640x360 and baseline sampling is
1fps. Two planned samples each tested the original input, neutral wording,
2fps, and added goalmouth magnifications. Doubling frames and magnifying did not
remove the false goal. Neutral wording was inconsistent; schema-rejected raw
responses still contained goal claims and must not count as abstentions.

A neutral ball-localization task marked the ball ambiguous in all three disputed
frames, while event narration confidently placed it inside the net. This supports
poorly calibrated visual interpretation. The confirmed system weakness is that
self-generated goal/line/net tags satisfy structural validation without independent
pixel verification, so both writers trust the same false event. Narrative completion
is an inference from outputs, not proven access to a model's internal reasoning.

No runtime fix was made: there is no verified single prompt/sampling adjustment
that fixes this failure. Future verification/abstention logic needs real-goal as
well as false-goal controls. Diagnostic harness: scripts/investigate_false_goals.py;
free audit by default, explicit --spend, no labels/expected outcome in runtime.
Nine small-model call attempts cost $0.0348588 under a combined $0.27 maximum;
no unsettled reservations, TTS, new footage, training or deployment. Validation:
372 offline tests passed, Ruff and mypy passed. Voice remains excluded and the
standing authorization for experiments at $1 or less still applies.

## Update: text evaluation trial completed, 9 November 2026

The user skipped voice, resumed text development, and asked for autonomous small
steps with progress updates. **Paid experiments with an explicit total cap of $1
or less are authorized; ask before anything above $1.** Do not split a larger
experiment to bypass this limit. This supersedes the older blanket paid pause and
the pending $1.20 question. Voice, audio generation and deployment remain paused.
The spending rule is also recorded in AGENTS.md.

Read [the text trial](TEXT_EVALUATION_TRIAL.md) and its
[complete record](../examples/evaluation-results/text-comparison-20261109.json)
next. The experiment's combined maximum was $1; reported token cost was
**$0.17030795**, with no unsettled reservations. There were 77 call attempts using
gpt-5.4-mini and one deliberate perception-only check using gpt-5.4-2026-03-05.
No TTS, fresh research, training, deployment or push occurred.

The shared-input text comparison is implemented in `scripts/compare_text.py` and
`src/commentary/replay/text_comparison.py`. `scripts/prepare_text_inputs.py` can
reuse saved inputs or make one bounded perception attempt per missing case. These
are offline full-context paths, not causal live measurements. Original demo paths
remain unchanged. Default modes make no paid calls; paid branches require --spend.

The v2 suite pilot had 12 usable perception inputs, 3 perception failures, 5/24
structurally complete writer arms and 19 failed arms. All 15 cases remain in the
report. Caller uncertainty validation differed between arms in that pilot; do not
use its completion counts as architecture superiority evidence. v3 aligned the
caller uncertainty/replay gate. v4 adds exact remaining-word feedback and stops
repeated caller requests after analyst abstention when there is no newer evidence.
A final paid dribble check produced only Goal! from both arms: it fits the supplied
timing, but omits the sequence and supplies no analysis. No full suite was rerun.

Both completed cross/header outputs falsely announced a goal because the shared
observer fabricated a confirmed goal with line/net cue tags. Coding-agent source
review contradicted that outcome. The larger existing model also invented a goal
with the same prompt/source/window/sampler, so no production model default changed.
There is **no demonstrated multi-agent quality advantage and no accepted candidate**.

Seven historical inputs remain in examples/text-comparison/inputs.json. A separate
prepared-inputs.json includes five new bundles (12 total); their perception errors
are preserved, not corrected from labels. Corner, foul-decision and offside remain
failed. The 15-case worksheet now contains integration outcomes and limited agent
semantic review, explicitly not independent human approval. Original reference
labels remain unreviewed. Local ignored runs retain raw providers, traces and media;
portable summaries, inputs, hashes and failures are committed.

Final validation: **370 offline tests passed**, Ruff and mypy passed. Regression tests use
scripted providers only. Next priority is false-goal perception and excessive late
support using existing inspection tools, then the three failed perception cases.
Do not feed target outcomes into runtime prompts, promote fabricated cue tags, or
claim timing-only passes establish good commentary.

This update supersedes older text-work status below while preserving the voice
pause and historical findings.

Read this first in a new session. This snapshot supersedes older status prose
where it conflicts with the recent recorded-demo work. The user requested this
handoff, not more implementation or audio generation.

## User intent and pause

Build an impressive but understandable AI engineering portfolio project: a real
multi-agent football commentary system, grounded research and perception,
evaluation, and measurable improvement using existing models. Keep the architecture
small enough to explain clearly in interviews.

**The user paused the immediate commentary/voice work. Keep it paused unless they
explicitly resume it.** They asked about the broader roadmap and are moving to a
new session. Do not generate more audio, launch paid experiments, deploy, or treat
the latest listening candidate as approved simply because a new session starts.
Read-only inspection and planning for the broader roadmap can proceed when asked.

Recent feedback, in order:

1. The original 20-second agent demo sounded poor across wording, emotion/voice,
   and timing, despite recognizing events and Kane correctly.
2. A revised candidate was "better" but sounded like disconnected sentences,
   not coherent ongoing commentary.
3. A continuity candidate was produced. The next feedback was doubt that these
   were the intended ElevenLabs voices. Its subjective quality is not accepted.
4. Voice investigation was paused. The intended voice names remain unanswered.

Do not equate passing tests, citations or on-cue audio with good commentary. The
previous assistant repeatedly overemphasized those checks without listening.

## Workspace and rules

- Canonical repo: `/Users/Aarsh/Developer/commentary`.
- Branch: `feature/replay-agents`; implementation HEAD before this handoff:
  `69ad473`. Working tree was clean when handoff work began.
- Work is committed locally. These recent changes were not pushed or deployed.
- Commit completed working changes regularly, short human messages, no co-author
  credits or `Co-Authored-By` trailers.
- Read `AGENTS.md`. No training, fine-tuning, or RL. The earlier RL idea was
  superseded by the user's explicit existing-models-only constraint.
- Regression tests use scripted backends, never paid model/TTS calls. Use cheap
  models for short deliberate development checks, hard dollar/call limits, no
  automatic retries. Larger models are only deliberate quality comparisons.
- Keep the selected 15 clips. Do not acquire broad datasets, request SoccerNet
  access, or expand infrastructure without an actual coverage gap.
- References, labels and target commentary stay outside runtime agent inputs.
- Use gstack `/browse` for web/browser QA, binary:
  `/Users/Aarsh/.agents/skills/gstack/browse/dist/browse`.
- Do not assume old subagent entries are active workers. Inspect their state if
  the user requests delegation; no new worker was launched in the recent work.
- `.env` contains credentials. Never print API keys or commit the file.

## What exists

The core is Python, LangGraph role/tool loops, asyncio coordination, and Pydantic
contracts. This is not a new framework migration.

- Researcher prepares cited, dated facts and squads, or loads a saved fixture-bound
  brief. The England squad was freshly researched and all 26 entries checked
  against the official dated squad-number article. Unverified France context was
  excluded. Broad pre-match statistical usefulness still needs evaluation.
- Observer recognizes typed actions from sampled frames, can request a bounded
  frame/crop inspection, and uses a research-roster tool for player identities.
  The model reads/proposes identity; Python validates, not deterministically
  selects a player from a shirt number. Speakers only see accepted names.
- Caller and analyst have separate prompts, tools, state, and model choices.
  They can speak, request permitted context, or abstain.
- A deterministic coordinator owns clocks, deadlines, visibility, turn scheduling,
  stale-result rejection and goal priority. Do not describe it as an LLM supervisor.
- A bounded reviewer can reject a draft and request at most one rewrite.
  Recorded preparation now enables it; live-oriented replay normally leaves it off.
- Saved local traces and an evaluator exist. Reviewer acceptance is not an
  independent factual or naturalness evaluation.

There are three distinct experiences:

1. **Five curated showcases:** Di María–France, Messi–Bayern, Arsenal–Norwich team
   goal, Ronaldo–Spain free kick, Ferran Torres–Betis. Scripts are manually edited
   and labelled accordingly. Plans: `examples/demo-showcase/`. They are not raw
   automatic baseline output.
2. **Causal replay core and saved execution traces:** advancing replay clock,
   independent agents, deadlines, cancellations and simulated speech occupancy.
   The old "27-second" penalty trace actually contains about 7 seconds of footage
   plus processing waits. UI now distinguishes the execution clock from source
   time. These traces do not establish real streaming speech latency.
3. **Prepared recorded agent demo:** full-context short-video vision pass and
   agent dialogue prepared before playback; actual custom-voice speech and captions.
   Source-time filters apply to speaking snapshots, but full-context vision and
   reserved future event slots mean this is not a causal live system.

The browser `/live` experiment also exists. Do not imply these separate paths
already form one fully integrated real-time broadcast system.

## Recent commits and code

| Commit | Change |
| --- | --- |
| `c32fe9a` | Connect researched squads to commentary |
| `fe1819e` | Full recorded agent conversation and source-time player |
| `6261026` | Broadcast phrasing, bounded draft review, spacing, speech context |
| `ca928f8` | Continuous caller performance, run-up slot, measured analyst handoff |
| `69ad473` | Save voice IDs/provider request IDs/audio hashes in future grouped audits |

Main files:

- `src/commentary/replay/recorded.py`: full-sequence recorded observer, batched
  agent identity resolution, bounded source-timed speaking turns. Identity
  proposals are batched in threes. Conservative normalization only weakens an
  invalid confirmed/unknown scene action; it does not promote a goal.
- `src/commentary/replay/agents.py`: role prompts, isolated LangGraph turns,
  tools, evidence checks, broadcast cues and optional review/revise branch.
- `src/commentary/replay/identity.py`: lookup stage is a small standalone schema,
  separate from identity proposals. This fixed concatenated/oversized outputs.
- `scripts/prepare_agent_demo.py`: free dry-run by default; `--spend` opts into
  bounded calls, optional cached vision, source SHA/window checks, provider-response
  capture before parsing, incremental outputs and final cost/error report.
- `scripts/synthesize_demo.py`: explicit `--spend`, distinct configured voices,
  character cap, saved voice settings, optional speech model and `--continuous`.
- `scripts/speech_blocks.py`: one timestamped ElevenLabs request per consecutive
  speaker block, exact character alignment checked, cuts only between phrases,
  reusable per-line WAV manifest. Provider response is saved before validation;
  malformed alignment stops without a paid retry.
- `scripts/render_demo.py`: local measured audio render/captions; optional
  `--response-gap 0.25` places analyst after actual caller audio, never before all
  cited observation/support timestamps. Caller cues remain anchored. No paid calls.
- `web/app/demo/`: prepared player, original curated videos, architecture and
  execution trace UI. `web/public/commentary-comparison.html` is a separate
  listening comparison, not a promoted main-demo replacement.

## Current local listening artifacts

Source: `clips/action-sources/england-france-penalties.mp4`, source seconds 42–62
(20 seconds). It has **no audio track**. No stadium ambience has been added.
Accepted underlying vision initially saw 24 seconds; the last four seconds cut
to another attack and were excluded from the final plan. This does not change
the separate 15-clip evaluation manifest.

| Version | Tracked evidence | Local ignored video |
| --- | --- | --- |
| Initial agent recording, rejected for quality | `examples/recorded-agent-demo/`, public `prepared-penalty.trace.json` | `web/public/soccer-commentary/prepared-penalty-agents.mp4` |
| Naturalness candidate, user said better but disconnected | `examples/recorded-agent-demo/naturalness/` | `web/public/soccer-commentary/natural-penalty-preview.mp4` |
| Latest continuity candidate, voice identity disputed | `examples/recorded-agent-demo/continuity/` | `web/public/soccer-commentary/continuous-penalty-preview.mp4` |

Latest measured placement:

| Start–end | Speaker | Words |
| --- | --- | --- |
| 1.00–2.01 | Caller | England drive on. |
| 3.00–5.19 | Caller | Penalty to England! They've got the big decision. |
| 9.00–9.54 | Caller | Kane... |
| 10.00–11.18 | Caller | Kane scores! |
| 11.43–14.68 | Analyst | He sends the keeper the wrong way, keeping it low and precise. |

Caller phrases share one generated performance, cut by verified provider
alignment and placed at event cues. Analyst recording is reused from the prior
candidate. Pauses remain during setup and after analysis; this is not continuous
speech over the entire 20 seconds. No subjective listening success is claimed.

Latest local run: `runs/recorded-agents/penalty-continuity-v2/`. Prior candidate:
`runs/recorded-agents/penalty-natural-quality-v2/`. These ignored directories retain
audio, manifests, provider output, alignment, traces and render sidecars. Some
one-off experiment drivers were in `/tmp`; do not rely on their persistence.
Tracked plans/traces and reusable scripts are the durable handoff.

All media and most `runs/` artifacts are git-ignored. They exist on this machine,
but a fresh clone only has plans, transcripts, traces and reports. Do not delete
or re-synthesize them unnecessarily. Older `prepared-penalty.mp4` is obsolete;
the current main player uses `prepared-penalty-agents.mp4`.

## Unresolved ElevenLabs identity investigation — paused

Current `.env` voice IDs (not API keys):

- Caller/lead: `7kZjjorO6NZ82NBwcvgV`.
- Analyst/supporting: `8aumr1JnqELHfstCHQOk`.
- Both the explicit CALLER/ANALYST variables and LEAD/SUPPORTING aliases agree;
  there were no shell voice overrides during the check.
- Speech model: `eleven_flash_v2_5`; no request-level voice-setting overrides.

Earlier initial-agent and naturalness audits explicitly recorded these IDs,
as did the older curated Di María demo. The final video demonstrably uses the
saved speech files: decoded waveform comparison matched caller input at 1.0
correlation and analyst at about 0.955 after compressed rendering. The complete
manifest means no macOS `say` fallback was used. These facts do **not** establish
that the stored IDs are the two voices the user intended in their account.

The latest grouped caller audit accidentally omitted its voice ID. `69ad473`
fixes future audits, including provider request ID and original-audio hash.
Do not retroactively backfill the old audit as if the provider verified it.

Read-only ElevenLabs requests returned 401:

- Voice details: key lacks `voices_read`.
- Speech history: key lacks `speech_history_read`.
- An earlier subscription read lacked `user_read`; TTS still worked.

The user was asked for the two intended voice names but did not answer before
pausing. If they resume this investigation, compare the actual account voice
names/IDs through authorized account access. Do not insist these must be the
right voices, switch IDs speculatively, ask them to paste keys, or generate new
audio merely to investigate identity.

## Evaluation state and bigger roadmap

Selection is `examples/action-suite.json`: 15 short development windows, 112.5
seconds total. Includes routine play, dribbling, tackle, interception, cross/header,
shot/save, missed and scored penalties, goal/reaction, corner, free kick, throw-in,
penalty award, red card and offside/VAR. Public local footage is already available.
Labels are sparse drafts with approximate timing, **not human-approved ground
truth**. All selected cases currently belong to development; no held-out general
accuracy claim. Related same-match windows are not independent samples.

Existing reports in `examples/evaluation-results/` include action smoke,
recognition, vision comparison, naming iterations, agent-owned identity and fresh
researched handoff. Read those before paying to repeat an experiment. The prepared
20-second voice example is not a successful run of the full suite.

`scripts/evaluate_replay.py` already measures event/action/timestamp matches,
citation coverage and availability, annotated identity agreement, scheduling,
simulated speech balance and trace diagnostics. It does not yet independently
judge semantic agreement with broadcaster commentary, tactical usefulness,
natural dialogue, pronunciation or emotional delivery. Those are unfinished
evaluation work, not properties proven by the current scorecard.

Remaining work, in a practical order when the user resumes broader development:

1. **Review evaluation references and define the acceptance rubric.** Separate
   recognition/naming, grounded spoken meaning, timing, useful analysis, naturalness
   and voice quality. Curate aligned actual-commentary references where available;
   semantic agreement does not mean copying the broadcaster's exact words. Keep
   every reference outside runtime inputs. Account for correct abstention and
   silence gaming. Split matches before any optimization/held-out claims.
2. **Baseline versus multi-agent comparison.** Compare the original automatic
   joint-writer path with the agents on identical clips and fixed budgets/configs.
   Curated manually edited showcase scripts cannot stand in for automatic baseline
   outputs. Reuse perception where appropriate and report that choice.
3. **Offline prompt improvement experiment.** GEPA was researched as a candidate,
   not installed/integrated as a completed optimization loop. Version role prompts,
   give development failures/feedback to a bounded optimizer, select on validation,
   keep test cases untouched, run regressions and promote manually. No weights,
   fine-tuning, RL or live self-editing. In-run reviewer revisions are separate
   from this across-run optimization experiment.
4. **Temporal play understanding.** Counterattacks, one-twos, switches, through
   balls, cutbacks and pressing require linked events/participants/possession
   across windows. This layer is not implemented; don't claim tactical recognition
   from an isolated action tag or plausible analyst sentence. Add only after
   establishing action quality on the small existing suite.
5. **Finish researcher usefulness.** Research tools and squad identification work
   in inspected cases; broader cited player/team pre-match stats still need useful
   analyst integration, factual checking, date/competition-scope preservation and
   measured value. Research is context, never proof of an event or player in video.
6. **Actual live speech integration.** Cancellable TTS/audio playback, measured
   event-to-audio delay, interruption and backpressure. Replay scheduling already
   exists, but its speech occupancy is simulated. Prepared full-context recordings
   do not establish causal real-time capability. LiveKit was a later media-adapter
   candidate, Pipecat an alternative; neither is needed merely for framework count.
7. **Portfolio evidence and release.** Update the architecture diagram and
   explanation to match implemented roles and distinguish recorded/live paths.
   Add honest comparisons, traces, cost/latency and failure analysis. Optional
   LangSmith experiment UI can complement portable local traces. Publish only a
   selected, reviewed candidate; no deployment was done in this recent work.

Research recommendation remains **LangGraph + asyncio + Pydantic**, optional
LangSmith and bounded standalone GEPA later. Do not migrate to CrewAI/Pydantic AI
or add services simply to sound agentic. The portfolio value is clear decisions,
tool use, evidence controls, coordination and demonstrated improvements.

## Validation, spend and resuming locally

Latest full offline run during handoff: **345 tests passed**. Mypy and Ruff passed
before the latest two-file audit fix; that fix also passed its 13 targeted tests.
Web lint/build passed during the recorded-demo work. Mobile comparison page was
checked at 375 px with no horizontal overflow and all three 20-second videos
loaded. No recent paid benchmark or listening test establishes broad quality.

Commands:

```bash
uv run pytest -q
uv run mypy
uv run ruff check src scripts/synthesize_demo.py scripts/speech_blocks.py scripts/render_demo.py tests
npm run lint --prefix web
npm run build --prefix web
npm run start --prefix web -- --port 3000
```

Local URLs: `/demo`, `/commentary-comparison.html#continuity`, `/live`.
A production server was running on 127.0.0.1:3000; check before starting another.
Do not build over a running production server: replacing `.next` caused a transient
500 until restart. Next's production server also needed restart to discover newly
added public media. Local processes/session IDs need not survive the new session.

Recent paid work only (not lifetime project totals):

- Initial full recording experiments: about $0.26 models and 91 voice credits.
- Naturalness iterations: $0.05472875 models and 87 voice credits.
- Continuity: known $0.0066885 models, plus one failed attempt bounded at $0.01
  whose exact usage was not retained; 44 voice credits. Analyst audio reused.

Some failed experiments did not save complete usage. Preserve that uncertainty;
do not invent precise totals. Reusable preparation now captures raw provider
responses before structured parsing. Avoid one-off drivers that lose audits when
exceptions occur, and don't repeat full-suite paid runs for routine code changes.

Read next: `docs/EVALUATION.md`, `docs/RESEARCH.md`, `docs/REPLAY.md`,
`docs/ACTION_SUITE.md`, `docs/RESEARCH_HANDOFF.md`, and the three recorded-agent
artifact directories. `docs/PORTFOLIO.md` contains older status descriptions that
need updating; in particular, do not repeat its stale blanket statements about
absent audio/research as the status of the new prepared player. Check actual CLI
defaults and recorded model IDs rather than treating old documentation as current.
