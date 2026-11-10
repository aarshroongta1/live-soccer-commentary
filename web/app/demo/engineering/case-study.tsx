"use client";

import { useState } from "react";
import diagrams from "@/data/engineering-diagrams.json";
import MermaidDiagram from "./mermaid-diagram";
import "../demo.css";
import "../showcase.css";
import "./engineering.css";

const skills = [
  { name: "Multimodal inference", text: <>Used <code>OpenCV</code> to sample timestamped frames and a vision model to propose actions, uncertainty and supporting evidence.</> },
  { name: "Agent orchestration", text: <><code>LangGraph</code> runs separate caller and analyst prompts with bounded tool loops. Agents choose to speak, request context or abstain.</> },
  { name: "Structured outputs & grounding", text: <><code>Pydantic</code> validates schemas and references. Evidence cutoffs, role permissions and word budgets constrain each proposed line.</> },
  { name: "Tool-augmented generation", text: <>Read-only <code>recent_events</code>, <code>player_context</code> and analyst <code>research</code> tools expose scoped context. Cited squads support identity, not proof of an action.</> },
  { name: "Async coordination", text: <>The separate replay prototype uses <code>asyncio</code> deadlines, cancellation, goal priority and stale-result rejection. The featured videos are prepared offline.</> },
  { name: "LLM evaluation", text: <>A fixed 15-clip development suite checks perception, identity, grounding, timing and dialogue. Regression tests use scripted backends without paid calls.</> },
  { name: "Observability & cost control", text: <>JSON traces retain model outputs, tool decisions, rejected turns and token costs. Call limits and budget reservations bound paid experiments.</> },
];

export default function EngineeringCaseStudy() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [expanded, setExpanded] = useState(false);
  return <div className="demo-page showcase-page engineering-page" data-theme={theme}>
    <main className="engineering-content">
      <nav className="showcase-nav" aria-label="Project navigation">
        <a className="showcase-author" href="https://www.aarshroongta.com">Aarsh Roongta<span aria-hidden="true">.</span></a>
        <div><a href="/demo">Watch demo</a><a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">GitHub ↗</a><button className="showcase-theme" type="button" aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`} aria-pressed={theme === "dark"} onClick={() => setTheme(theme === "light" ? "dark" : "light")}>{theme === "light" ? "◐" : "◑"}</button></div>
      </nav>
      <header className="engineering-heading">
        <p><a href="/demo">Football commentary</a><span aria-hidden="true">/</span>Engineering</p>
        <h1>How I built it<span aria-hidden="true">.</span></h1>
        <p>A vision model reads the play. Two specialist agents write commentary. Code checks evidence and timing.</p>
      </header>

      <MermaidDiagram diagram={diagrams.overview} theme={theme} showSource={false} responsive />
      <section className="engineering-highlights" aria-label="Key engineering skills">
        <article><h2>Multimodal inference</h2><p>OpenCV + vision models turn footage into timestamped evidence.</p></article>
        <article><h2>Multi-agent orchestration</h2><p>LangGraph roles use scoped tools; Pydantic validates their proposed lines.</p></article>
        <article><h2>Evaluation & observability</h2><p>15 development clips, saved decision traces, and hard cost limits.</p></article>
      </section>
      <section className="engineering-result" aria-labelledby="result-title">
        <div><span>One evaluation-driven change</span><h2 id="result-title">Restored the caller’s goal reaction.</h2></div>
        <p>A reviewer blocked “Kane scores!” Disabling it restored the caller-to-analyst handoff.<small>One saved example; the first call also started later.</small></p>
      </section>

      <details className="engineering-explore" onToggle={(event) => setExpanded(event.currentTarget.open)}>
        <summary>Explore further <span>Architecture, tradeoffs and resume summary</span></summary>
        {expanded && <div className="engineering-expanded">
          <section className="engineering-section" aria-labelledby="architecture-title">
            <h2 id="architecture-title">Full system architecture</h2>
            <MermaidDiagram diagram={diagrams.system} theme={theme} />
            <p className="engineering-note">Recorded preparation sees the short clip before playback. England reuses saved perception and squad research; Arsenal and Argentina use fresh perception with disclosed light edits.</p>
          </section>

      <section className="engineering-section" aria-labelledby="skills-title">
        <h2 id="skills-title">The engineering behind it</h2>
        <dl className="engineering-skills">{skills.map((skill) => <div key={skill.name}><dt>{skill.name}</dt><dd>{skill.text}</dd></div>)}</dl>
      </section>

      <section className="engineering-section" aria-labelledby="agent-turn-title">
        <h2 id="agent-turn-title">Inside an agent turn</h2>
        <p>A role receives only its allowed evidence and context. LangGraph routes the action; application code executes tools and validates speech.</p>
        <MermaidDiagram diagram={diagrams.turn} theme={theme} compact />
        <p className="engineering-note">Tool use follows a typed application action protocol. The optional model reviewer is disabled in the selected runs; deterministic checks remain enabled.</p>
      </section>

      <section className="engineering-section" aria-labelledby="decisions-title">
        <h2 id="decisions-title">What evaluation changed</h2>
        <div className="engineering-decisions">
          <article><h3>Timing became a contract</h3><p>Short clips exposed drafts that could not fit after their evidence arrived. Per-turn word budgets and clearer rejection feedback made this constraint explicit.</p></article>
          <article><h3>The reviewer needed evaluation too</h3><p>In one saved penalty pair, a reviewer rejected “Kane scores!” twice. Disabling it restored the caller-to-analyst handoff. Calls fell from 17 to 9, but the first call also moved from 1s to 3s.</p></article>
          <article><h3>Shared evidence is a shared failure point</h3><p>A false visual goal reached both writers. Valid JSON and citations did not establish factual correctness. I kept the failed traces and separated perception failures from writer failures.</p></article>
        </div>
        <p className="engineering-note">The pilot had 12 usable perception inputs; 5 of 24 writer attempts passed structural checks. These are development diagnostics with draft labels, not a held-out benchmark. The single penalty pair does not establish general quality or cost improvement.</p>
        <div className="engineering-links"><a href="/demo/engineering/runs">Inspect the saved runs</a><a href="/soccer-commentary/evaluation-notes.txt" target="_blank" rel="noreferrer">Read evaluation notes</a></div>
      </section>

      <section className="engineering-section engineering-resume" aria-labelledby="resume-title">
        <h2 id="resume-title">Project summary</h2>
        <blockquote>Built a Python/LangGraph football commentary prototype with multimodal perception, tool-using caller and analyst agents, timestamped evidence checks, and a 15-clip evaluation workflow with reproducible failure traces.</blockquote>
        <p className="engineering-note">The demo uses existing models and precomputed captions. The replay prototype simulates speech timing; measured live speech latency is outside this demonstration.</p>
      </section>
        </div>}
      </details>
      <footer className="showcase-footer"><a href="/demo">Back to the demo</a><a href="/soccer-commentary/engineering-overview.md" download>Download the README-style overview</a><a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">Source code ↗</a></footer>
    </main>
  </div>;
}
