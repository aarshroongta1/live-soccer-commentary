"use client";

import { useEffect, useRef, useState } from "react";
import AgentRuns, { type AgentRun } from "./agent-runs";
import CommentaryArchitecture from "./architecture";
import "./demo.css";

export type DemoClip = {
  id: string;
  title: string;
  subtitle: string;
  src: string | null;
  poster: string | null;
  duration_s: number;
  kind: "automatic" | "lightly-edited" | "curated";
  note: string;
  lines: { at_s: number; voice: string; text: string }[];
};

type Preparation = {
  tracePath: string;
  editPath?: string;
  editSummary?: string[];
  costNote?: string;
  turnTimeNote?: string;
  method: string[];
  generation: { model: string; calls: number; cost_usd: number; rejected_turns: number };
  observations: { id: string; at_s: number; description: string }[];
  turns: { role: string; at_s: number; status: string; detail: string }[];
};

export type Iteration = {
  title: string;
  summary: string;
  scope: string;
  before: { clip: DemoClip; preparation: Preparation };
  measurements: { label: string; before: string; after: string }[];
  rejectedDraft: string;
  reviewerFeedback: string;
  limitation: string;
};

const timestamp = (seconds: number) => `0:${Math.floor(seconds).toString().padStart(2, "0")}`;
const provenance = (kind: DemoClip["kind"]) => kind === "automatic" ? "Unedited AI text" : kind === "lightly-edited" ? "Lightly edited AI" : "Edited showcase";
const inspectorTabs = ["Transcript", "Evidence", "Run"] as const;
type InspectorTab = typeof inspectorTabs[number];

function ClipPlayer({ clip, preparation }: { clip: DemoClip; preparation: Preparation }) {
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  const [selected, setSelected] = useState(-1);
  const [failed, setFailed] = useState(false);
  const [tab, setTab] = useState<InspectorTab>("Transcript");
  const [sourceTime, setSourceTime] = useState(0);
  const available = Boolean(clip.src) && !failed;

  useEffect(() => {
    const player = video.current;
    return () => { player?.pause(); };
  }, []);

  function seek(at: number) {
    if (!video.current || !available) return;
    if (video.current.readyState === 0) pendingSeek.current = at;
    else video.current.currentTime = at;
    setSourceTime(at);
    setSelected(clip.lines.findLastIndex((line) => line.at_s <= at));
  }

  return (
    <div className="demo-workspace">
      <div className="demo-workspace-bar">
        <div><span className="demo-saved-dot" aria-hidden="true" />Recorded playback<span className="demo-workspace-clock">{timestamp(sourceTime)} / {timestamp(clip.duration_s)}</span></div>
        <a href={preparation.tracePath} download>Download trace <span aria-hidden="true">↗</span></a>
      </div>
      <div className="demo-player-layout">
        <div className="demo-video-column">
          {available ? (
            <video
              ref={video} src={clip.src!} poster={clip.poster ?? undefined}
              controls muted playsInline preload="metadata" data-commentary-player
              aria-label={`${clip.title}: silent video with commentary captions`}
              onPlay={(event) => {
                document.querySelectorAll<HTMLVideoElement>("video[data-commentary-player]").forEach((player) => {
                  if (player !== event.currentTarget) player.pause();
                });
              }}
              onTimeUpdate={(event) => {
                const at = event.currentTarget.currentTime;
                setSourceTime(at);
                setSelected(clip.lines.findLastIndex((line) => line.at_s <= at));
              }}
              onLoadedMetadata={(event) => {
                if (pendingSeek.current !== null) {
                  event.currentTarget.currentTime = pendingSeek.current;
                  pendingSeek.current = null;
                }
              }}
              onError={() => setFailed(true)}
            >Your browser cannot play the video. Read the transcript alongside it.</video>
          ) : (
            <div className="demo-missing" role="status">
              <h3>Video unavailable</h3>
              <p>The saved transcript and evidence are still available. See the repository for local media setup.</p>
            </div>
          )}
          <div className="demo-player-footer">
            <span>{clip.duration_s}s footage <span aria-hidden="true">/</span> Captions <span aria-hidden="true">/</span> No audio</span>
            {clip.src && <a href={clip.src} download>Download clip</a>}
          </div>
        </div>
        <section className="demo-transcript-panel" aria-label={`${clip.title} inspector`}>
          <div className="demo-inspector-tabs" role="tablist" aria-label="Inspect the saved run">
            {inspectorTabs.map((name, index) => (
              <button key={name} id={`${clip.id}-tab-${name}`} type="button" role="tab"
                aria-selected={tab === name} aria-controls={`${clip.id}-panel`} tabIndex={tab === name ? 0 : -1}
                onClick={() => setTab(name)} onKeyDown={(event) => {
                  const next = event.key === "ArrowRight" ? (index + 1) % 3 : event.key === "ArrowLeft" ? (index + 2) % 3 : event.key === "Home" ? 0 : event.key === "End" ? 2 : null;
                  if (next === null) return;
                  event.preventDefault();
                  setTab(inspectorTabs[next]);
                  document.getElementById(`${clip.id}-tab-${inspectorTabs[next]}`)?.focus();
                }}>{name}{name === "Evidence" && <span>{preparation.observations.length}</span>}</button>
            ))}
          </div>
          <div className="demo-inspector-body" id={`${clip.id}-panel`} role="tabpanel" aria-labelledby={`${clip.id}-tab-${tab}`} tabIndex={0}>
            {tab === "Transcript" && <>
              <p className="demo-inspector-hint">Choose a line to seek the footage.</p>
              <ol className="demo-lines">
                {clip.lines.map((line, index) => (
                  <li key={`${clip.id}-${index}`}>
                    <button type="button" aria-current={selected === index ? "true" : undefined} onClick={() => seek(line.at_s)}>
                      <time>{timestamp(line.at_s)}</time>
                      <span className={`demo-voice demo-voice-${line.voice}`}>{line.voice === "caller" ? "Caller" : "Analyst"}</span>
                      <span>{line.text}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </>}
            {tab === "Evidence" && <>
              <p className="demo-inspector-hint">Saved model observations. Select a timestamp to inspect the footage; these are not verified labels.</p>
              <ol className="demo-observations">
                {preparation.observations.map((event) => <li key={event.id}>
                  <div><button type="button" onClick={() => seek(event.at_s)} aria-label={`View observation ${event.id} at ${timestamp(event.at_s)}`}>{timestamp(event.at_s)}</button><code>{event.id}</code></div>
                  <p>{event.description}</p>
                </li>)}
              </ol>
            </>}
            {tab === "Run" && <div className="demo-run-metadata">
              <p className="demo-inspector-hint">Generation finished before playback. No model calls run on this page.</p>
              <dl>
                <div><dt>Models</dt><dd><code>{preparation.generation.model}</code></dd></div>
                <div><dt>Model calls</dt><dd>{preparation.generation.calls}</dd></div>
                <div><dt>Reported token cost</dt><dd>${preparation.generation.cost_usd.toFixed(4)}</dd></div>
                <div><dt>Turns rejected by checks</dt><dd>{preparation.generation.rejected_turns}</dd></div>
                <div><dt>Text provenance</dt><dd>{provenance(clip.kind)}</dd></div>
              </dl>
              <p>{preparation.costNote ?? "Saved perception and research costs are excluded. This is selected development output."}</p>
              <a href={preparation.tracePath} download>Download full run JSON</a>
            </div>}
          </div>
        </section>
      </div>
    </div>
  );
}

export default function DemoExperience({ clips, runs, preparations, iteration }: {
  clips: DemoClip[]; runs: AgentRun[]; preparations: Record<string, Preparation>; iteration: Iteration;
}) {
  const [activeId, setActiveId] = useState(clips[0].id);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const active = clips.find((clip) => clip.id === activeId)!;
  const [version, setVersion] = useState<"before" | "after">("after");
  const hasIteration = active.id === "portfolio-penalty";
  const showBefore = hasIteration && version === "before";
  const displayed = showBefore ? iteration.before.clip : active;
  const visiblePreparation = showBefore ? iteration.before.preparation : preparations[active.id];
  return (
    <div className="demo-page" data-theme={theme}>
      <a className="demo-skip-link" href="#demos">Skip to demo</a>
      <div className="demo-shell">
        <aside className="demo-rail" aria-label="Project navigation">
          <a className="demo-author" href="https://www.aarshroongta.com">Aarsh <br />Roongta<span aria-hidden="true">.</span></a>
          <a className="demo-back" href="/demo">Back to demo <span aria-hidden="true">↗</span></a>
          <nav className="demo-rail-nav" aria-label="On this page">
            <a href="#demos">Demo</a><a href="#pipeline">Pipeline</a><a href="#evaluation">Evaluation</a>
          </nav>
          <div className="demo-rail-links">
            <a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a>
            <a href="/soccer-commentary/demo-walkthrough.txt" download>Walkthrough</a>
          </div>
          <button className="demo-theme-toggle" type="button" aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`} aria-pressed={theme === "dark"} onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
            <span aria-hidden="true">{theme === "light" ? "◐" : "◑"}</span> {theme === "light" ? "Dark theme" : "Light theme"}
          </button>
          <p className="demo-rail-note">A project in multimodal<br />AI and agent coordination.</p>
        </aside>
        <main className="demo-content">
          <header className="demo-heading">
            <p className="demo-project-path"><a href="https://www.aarshroongta.com/projects">projects</a><span aria-hidden="true">/</span>football commentary</p>
            <h1>Football commentary agents<span aria-hidden="true">.</span></h1>
            <p>Vision-grounded commentary from a caller and analyst, coordinated by code.</p>
            <ul className="demo-stack" aria-label="Project technologies">{["Python", "LangGraph", "OpenCV", "Next.js"].map((name) => <li key={name}>{name}</li>)}</ul>
          </header>

          <section id="demos" className="demo-selected" aria-labelledby="demos-title">
            <div className="demo-section-heading"><h2 id="demos-title">Watch a saved run</h2><span>Prepared before playback</span></div>
            <div className="demo-clip-options" role="group" aria-label="Choose a demo clip">
              {clips.map((clip) => (
                <button key={clip.id} type="button" aria-pressed={clip.id === activeId} onClick={() => setActiveId(clip.id)}>
                  <strong>{clip.title}</strong><span className="demo-clip-duration">{clip.duration_s}s</span>
                  <span className={`demo-kind demo-kind-${clip.kind}`}>{provenance(clip.kind)}</span>
                </button>
              ))}
            </div>
            <div className="demo-selected-heading">
              <div><h3>{active.subtitle}</h3></div>
              {hasIteration && <div className="demo-version-controls" role="group" aria-label="Compare automatic versions">
                <button type="button" aria-label="Before: reviewer enabled" aria-pressed={version === "before"} onClick={() => setVersion("before")}>Before <span>reviewer on</span></button>
                <button type="button" aria-label="After: reviewer disabled" aria-pressed={version === "after"} onClick={() => setVersion("after")}>After <span>reviewer off</span></button>
              </div>}
            </div>
            <ClipPlayer key={displayed.id} clip={displayed} preparation={visiblePreparation} />
            <p className="demo-provenance"><span>{provenance(displayed.kind)}</span>{displayed.note}</p>
            {hasIteration && <details className="demo-iteration-result">
              <summary><span>Iteration notes</span> Why the caller’s goal reaction was missing</summary>
              <h3>{iteration.title}</h3><p>{iteration.summary}</p>
              <div className="demo-iteration-evidence">
                <div><h4>The rejected draft</h4><blockquote>{iteration.rejectedDraft}</blockquote><p>The reviewer requested an even shorter reaction. The rewrite allowance ran out, so the caller’s turn was dropped. Disabling that reviewer restored the caller’s reaction; deterministic checks remained enabled.</p></div>
                <table><caption>Observed in one matched development pair</caption><thead><tr><th scope="col">Measure</th><th scope="col">Before</th><th scope="col">After</th></tr></thead><tbody>{iteration.measurements.map((row) => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.before}</td><td>{row.after}</td></tr>)}</tbody></table>
              </div>
              <p className="demo-comparison-limit">{iteration.scope} {iteration.limitation}</p>
              <a href="/soccer-commentary/portfolio-iteration.json" download>Download the comparison and input hashes</a>
            </details>}
            {visiblePreparation.editSummary && <details className="demo-method demo-edit-notes">
              <summary><span>Editorial notes</span> See the exact light edits</summary>
              <ul>{visiblePreparation.editSummary.map((edit) => <li key={edit}>{edit}</li>)}</ul>
              <a href={visiblePreparation.editPath} download>Download edit record</a>
            </details>}
            <details className="demo-method demo-preparation-notes">
              <summary><span>Preparation notes</span> Inspect the full agent run</summary>
              <div className="demo-method-grid">
                <div><h3>From evidence to dialogue</h3><ul>{visiblePreparation.method.map((step) => <li key={step}>{step}</li>)}</ul><p>{visiblePreparation.costNote ?? "Saved vision and research costs are excluded. This is selected development output."}</p><a href={visiblePreparation.tracePath} download>Download the complete preparation trace</a></div>
                <div><h3>Source-time turns</h3>{visiblePreparation.turnTimeNote && <p>{visiblePreparation.turnTimeNote}</p>}<ol className="demo-turns">{visiblePreparation.turns.map((turn, i) => <li key={i}><time>{timestamp(turn.at_s)}</time><div><strong>{turn.role} / {turn.status}</strong><p>{turn.detail}</p></div></li>)}</ol></div>
              </div>
            </details>
          </section>

          <CommentaryArchitecture />

          <section id="evaluation" className="demo-evaluation" aria-labelledby="evaluation-title">
            <div className="demo-section-heading"><h2 id="evaluation-title">What evaluation changed</h2><span>15 development clips</span></div>
            <p>I checked recognition, factual grounding, timing and dialogue separately. The useful result was finding where the system broke, then making targeted changes.</p>
            <div className="demo-evaluation-records">
              <article><h3>Short clips exposed timing failures</h3><p>The pilot produced 12 usable perception inputs. Only 5 of 24 writer attempts passed structural checks. Later changes tightened word-budget feedback and role permissions.</p><code>timing + permission checks</code></article>
              <article><h3>Shared evidence can share mistakes</h3><p>A false goal reached both writers through the same visual observation. Citations and valid JSON did not prove the event happened. A larger model repeated the error.</p><code>grounding ≠ factual correctness</code></article>
              <article><h3>An extra reviewer can reject a good call</h3><p>In the penalty example, the reviewer rejected “Kane scores!” twice. Disabling it restored the caller-to-analyst handoff. Both outputs and the delayed first call are retained.</p><code>inspect the failure, test the change</code></article>
            </div>
            <p className="demo-evaluation-limit">Draft development labels, not a held-out benchmark. These selected clips and the one before/after pair do not demonstrate a general multi-agent quality advantage.</p>
            <div className="demo-document-links"><a href="/soccer-commentary/evaluation-notes.txt" target="_blank" rel="noreferrer">Evaluation notes</a><a href="/soccer-commentary/portfolio-release.json" download>All four fresh-run attempts + costs</a></div>
          </section>

          <details className="demo-replay-details"><summary><span>Further exploration</span> Open the causal replay prototype</summary><p>Separate saved runs show deadline handling, agent tools and abstention. Their execution clocks include processing waits; speech occupancy is simulated.</p><AgentRuns runs={runs} /></details>
          <footer className="demo-footer"><p>Built by <a href="https://www.aarshroongta.com">Aarsh Roongta</a>.</p><span>Existing models. Inspectable decisions. Recorded examples.</span><a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">Source on GitHub <span aria-hidden="true">↗</span></a></footer>
        </main>
      </div>
    </div>
  );
}
