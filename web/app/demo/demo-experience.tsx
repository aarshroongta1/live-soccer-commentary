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
  kind: "automatic" | "curated";
  note: string;
  lines: { at_s: number; voice: string; text: string }[];
};

type Preparation = {
  tracePath: string;
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

function ClipPlayer({ clip }: { clip: DemoClip }) {
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  const [selected, setSelected] = useState(-1);
  const [failed, setFailed] = useState(false);
  const available = Boolean(clip.src) && !failed;

  useEffect(() => {
    const player = video.current;
    return () => { player?.pause(); };
  }, []);

  return (
    <div className="demo-player-layout">
      <div>
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
            onTimeUpdate={(event) => setSelected(clip.lines.findLastIndex((line) => line.at_s <= event.currentTarget.currentTime))}
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
            <h3>The video is unavailable here.</h3>
            <p>The full transcript is still available. Local setup instructions are in the source repository.</p>
          </div>
        )}
        <div className="demo-player-footer">
          <span>{clip.duration_s} seconds · Captioned video · No audio</span>
          {clip.src && <a href={clip.src} download>Download clip</a>}
        </div>
        <p className="demo-provenance">{clip.note}</p>
      </div>
      <section className="demo-transcript-panel" aria-label={`${clip.title} transcript`}>
        <h3>Commentary</h3>
        <p>Choose a line to jump to that moment.</p>
        <ol className="demo-lines">
          {clip.lines.map((line, index) => (
            <li key={`${clip.id}-${index}`}>
              <button type="button" aria-current={selected === index ? "true" : undefined}
                onClick={() => {
                  setSelected(index);
                  if (video.current && available) {
                    if (video.current.readyState === 0) pendingSeek.current = line.at_s;
                    else video.current.currentTime = line.at_s;
                  }
                }}>
                <time>{timestamp(line.at_s)}</time>
                <span className={`demo-voice demo-voice-${line.voice}`}>{line.voice === "caller" ? "Caller" : "Analyst"}</span>
                <span>{line.text}</span>
              </button>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

export default function DemoExperience({ clips, runs, preparation, iteration }: {
  clips: DemoClip[]; runs: AgentRun[]; preparation: Preparation; iteration: Iteration;
}) {
  const [activeId, setActiveId] = useState(clips[0].id);
  const active = clips.find((clip) => clip.id === activeId)!;
  const [version, setVersion] = useState<"before" | "after">("after");
  const showBefore = active.kind === "automatic" && version === "before";
  const displayed = showBefore ? iteration.before.clip : active;
  const visiblePreparation = showBefore ? iteration.before.preparation : preparation;
  return (
    <main className="demo-page">
      <div className="demo-content">
        <nav className="demo-nav" aria-label="Project navigation">
          <a href="https://www.aarshroongta.com/projects">Aarsh Roongta / Projects</a>
          <a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">View source</a>
        </nav>
        <header className="demo-heading">
          <h1>Football commentary, built with agents.</h1>
          <p>Independent AI agents watch, research, call and explain the play.</p>
          <nav className="demo-section-nav" aria-label="On this page">
            <a href="#demos">Watch the demos</a>
            <a href="#demo-architecture-title">How it works</a>
            <a href="#evaluation">What I evaluated</a>
          </nav>
        </header>

        <section id="demos" className="demo-selected" aria-labelledby="demos-title">
          <div className="demo-section-heading"><h2 id="demos-title">Three moments from the project</h2><span>Recorded examples</span></div>
          <div className="demo-clip-options" role="group" aria-label="Choose a demo clip">
            {clips.map((clip) => (
              <button key={clip.id} type="button" aria-pressed={clip.id === activeId}
                onClick={() => setActiveId(clip.id)}>
                <span className={`demo-kind demo-kind-${clip.kind}`}>{clip.kind === "automatic" ? "Automatic dialogue" : "Edited showcase"}</span>
                <strong>{clip.title}</strong><span>{clip.subtitle}</span>
              </button>
            ))}
          </div>
          {active.kind === "automatic" && (
            <section className="demo-iteration" aria-labelledby="iteration-title">
              <h3 id="iteration-title">{iteration.title}</h3>
              <p>{iteration.scope}</p>
              <div className="demo-version-controls" role="group" aria-label="Compare automatic versions">
                <button type="button" aria-pressed={version === "before"} onClick={() => setVersion("before")}>Before: reviewer enabled</button>
                <button type="button" aria-pressed={version === "after"} onClick={() => setVersion("after")}>After: reviewer disabled</button>
              </div>
            </section>
          )}
          <ClipPlayer key={displayed.id} clip={displayed} />
          {active.kind === "automatic" && (
            <details className="demo-iteration-result" open>
              <summary>What changed after evaluating the first attempt?</summary>
              <p>{iteration.summary}</p>
              <div className="demo-iteration-evidence">
                <div><h4>A valid draft was rejected</h4>
                  <blockquote>{iteration.rejectedDraft}</blockquote>
                  <p>The reviewer still asked for a shorter reaction. The caller’s turn was dropped after its rewrite allowance was exhausted. Deterministic evidence, permission and timing checks remain enabled in the after run.</p>
                </div>
                <table><caption>Observed in this matched development pair</caption><thead><tr><th scope="col">Measure</th><th scope="col">Before</th><th scope="col">After</th></tr></thead>
                  <tbody>{iteration.measurements.map((row) => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.before}</td><td>{row.after}</td></tr>)}</tbody>
                </table>
              </div>
              <p className="demo-comparison-limit">{iteration.limitation}</p>
              <a href="/soccer-commentary/portfolio-iteration.json" download>Download the comparison record and input hashes</a>
            </details>
          )}
          {active.kind === "automatic" && (
            <details className="demo-method">
              <summary>Inspect this agent run</summary>
              <div className="demo-method-grid">
                <div>
                  <h3>From evidence to dialogue</h3>
                  <ul>{visiblePreparation.method.map((step) => <li key={step}>{step}</li>)}</ul>
                  <p>{visiblePreparation.generation.calls} new model calls · ${visiblePreparation.generation.cost_usd.toFixed(4)} in reported token cost · {visiblePreparation.generation.rejected_turns} turn(s) rejected by checks</p>
                  <p>Saved vision and research costs are excluded. This is selected development output.</p>
                  <a href={visiblePreparation.tracePath} download>Download the complete preparation trace</a>
                </div>
                <div>
                  <h3>Source-time turns</h3>
                  <ol className="demo-turns">{visiblePreparation.turns.map((turn, i) => (
                    <li key={i}><time>{timestamp(turn.at_s)}</time><div><strong>{turn.role} · {turn.status}</strong><p>{turn.detail}</p></div></li>
                  ))}</ol>
                </div>
              </div>
              <details className="demo-evidence"><summary>Visual observations used by the agents</summary>
                <ul>{visiblePreparation.observations.map((event) => <li key={event.id}><time>{timestamp(event.at_s)}</time> {event.description}</li>)}</ul>
              </details>
            </details>
          )}
        </section>

        <CommentaryArchitecture />

        <section id="evaluation" className="demo-evaluation" aria-labelledby="evaluation-title">
          <h2 id="evaluation-title">What I evaluated</h2>
          <p>Fifteen short development clips cover routine play, goals, saves and referee decisions. I kept recognition, factual grounding, timing and dialogue quality separate.</p>
          <div className="demo-evaluation-grid">
            <div><h3>Reproducible runs</h3><p>Saved inputs, model choices, tool decisions, rejected drafts and token costs make failures inspectable. Offline regression tests use scripted models.</p></div>
            <div><h3>A useful failure</h3><p>A false visual goal reached both writers through shared evidence. Valid citations alone could not establish factual correctness. A larger model repeated the error.</p></div>
            <div><h3>An honest comparison</h3><p>The text pilot had 12 usable inputs and only 5 of 24 writer attempts passed structural checks. It demonstrated no multi-agent quality advantage. A later timing fix fit a short call, but omitted the buildup.</p></div>
          </div>
          <p className="demo-evaluation-limit">These are development findings with draft labels, not a held-out accuracy benchmark. The edited showcases above illustrate the intended presentation; the automatic example shows actual generated wording.</p>
          <a href="/soccer-commentary/evaluation-notes.txt" target="_blank" rel="noreferrer">Read the evaluation notes</a>
        </section>

        <details className="demo-replay-details"><summary>Explore the causal replay prototype</summary>
          <p>Separate saved runs show deadline handling, agent tools and abstention. Their execution clocks include processing waits; speech occupancy is simulated.</p>
          <AgentRuns runs={runs} />
        </details>
        <footer className="demo-footer"><p>Built by Aarsh Roongta with Python, LangGraph, asyncio, Pydantic and Next.js.</p><a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">Code, traces and experiment records</a></footer>
      </div>
    </main>
  );
}
