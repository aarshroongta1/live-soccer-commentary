"use client";

import { useEffect, useRef, useState } from "react";

export type AgentRun = {
  id: string;
  title: string;
  src: string | null;
  summary: string;
  limitation: string;
  source_duration_s: number;
  duration_s: number;
  research_used: boolean;
  inspection: { frame_at_s: number; context_at_s: number[] } | null;
  lines: { id: string; role: string; text: string; start_s: number; end_s: number;
    interrupted: boolean; evidence_ids: string[]; origin: string }[];
  events: { id: string; time_s: number; kind: string; title: string; role: string | null;
    detail: string; source_at_s: number | null; observations: { id: string; at_s: number;
      description: string; action: { type: string; certainty: string } | null;
      actor?: { resolved_name: string | null; shirt_number: number | null;
        shirt_name: string | null; action_role: string } | null }[] }[];
  provenance: { path: string; sha256: string; speech_timing: string; edited: boolean; reuse_note?: string | null };
};
const seconds = (time: number) => `${time.toFixed(1)}s`;

function RunReview({ run }: { run: AgentRun }) {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);
  const [frame, setFrame] = useState<number | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const playbackOrigin = useRef({ at: 0, wall: 0 });
  const visible = run.events.filter((event) => event.time_s <= cursor);
  const active = visible.at(-1);
  const evidence = visible.flatMap((event) => event.observations);
  const lines = run.lines.filter((line) => line.start_s <= cursor);
  const sourceTime = Math.min(frame ?? cursor, Math.max(0, run.source_duration_s - 0.05));

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      const origin = playbackOrigin.current;
      const next = Math.min(origin.at + (performance.now() - origin.wall) / 1000, run.duration_s);
      setCursor(next);
      if (next >= run.duration_s) setPlaying(false);
    }, 50);
    return () => window.clearInterval(timer);
  }, [playing, run.duration_s]);

  useEffect(() => {
    const player = video.current;
    if (!player) return;
    if (Math.abs(player.currentTime - sourceTime) > 0.25) player.currentTime = sourceTime;
    if (playing && cursor < run.source_duration_s && cursor < run.duration_s) {
      void player.play().catch(() => { /* The timeline remains usable if playback is blocked. */ });
    } else player.pause();
  }, [playing, cursor, sourceTime, run.source_duration_s, run.duration_s]);

  useEffect(() => {
    const player = video.current;
    return () => player?.pause();
  }, []);

  function seek(time: number, evidenceTime: number | null = null) {
    setPlaying(false);
    setFrame(evidenceTime);
    setCursor(time);
  }

  const running = playing && cursor < run.duration_s;
  return (
    <>
      <p className="demo-run-summary">{run.summary}</p>
      <div className="demo-run-grid">
        <div>
          {run.src && !failed ? (
            <video ref={video} src={run.src} muted playsInline preload="metadata"
              aria-label={`${run.title}, muted source footage`}
              onError={() => setFailed(true)}
              onLoadedMetadata={() => { if (video.current) video.current.currentTime = sourceTime; }} />
          ) : (
            <p className="demo-missing">Source video isn’t available here. The saved trace and text still work.</p>
          )}
          <p className="demo-run-caption">
            Muted source footage · {seconds(run.source_duration_s)}
            {frame !== null && ` · Viewing evidence at ${seconds(frame)}`}
          </p>
          <div className="demo-run-controls">
            <button type="button" onClick={() => {
              setFrame(null);
              if (running) { setPlaying(false); return; }
              const next = cursor >= run.duration_s ? 0 : cursor;
              playbackOrigin.current = { at: next, wall: performance.now() };
              setCursor(next);
              setPlaying(true);
            }}>{running ? "Pause replay" : "Replay saved run"}</button>
            <button type="button" onClick={() => seek(run.duration_s)}>Show result</button>
            <button type="button" onClick={() => seek(0)}>Reset</button>
          </div>
          <label className="demo-run-clock">
            <span>Run clock <output>{seconds(cursor)} / {seconds(run.duration_s)}</output></span>
            <input type="range" min={0} max={run.duration_s} step={0.05} value={cursor}
              onChange={(event) => seek(Number(event.target.value))}
              aria-label="Seek saved run time" />
          </label>
          <p className="demo-run-caption">The run clock continues after the source clip ends. Processing delays are preserved.</p>
          <div className="demo-run-dialogue" aria-label="Saved commentary text">
            <h3>Commentary text</h3>
            {lines.length === 0 && <p>No call yet. Use “Show result” to skip the processing wait.</p>}
            {lines.map((line) => (
              <div key={line.id} className={`demo-run-line demo-run-line-${line.role}`}>
                <p className="demo-run-speaker">{line.role === "caller" ? "Caller" : "Analyst"} <time>{seconds(line.start_s)}</time></p>
                <p>{line.text}</p>
                <small>{line.origin}{line.interrupted ? " · Interrupted at run end" : ""}</small>
              </div>
            ))}
            {visible.filter((event) => event.kind === "abstained").map((event) => (
              <p className="demo-run-abstained" key={event.id}><strong>Analyst stayed quiet.</strong> {event.detail}</p>
            ))}
          </div>
        </div>
        <div className="demo-run-trace">
          <h3>Agent activity</h3>
          <ol aria-label="Saved agent activity timeline">
            {run.events.map((event) => (
              <li key={event.id} className={event.time_s > cursor ? "demo-run-pending" : ""}>
                <button type="button" onClick={() => seek(event.time_s)} aria-current={active?.id === event.id ? "step" : undefined}>
                  <time>{seconds(event.time_s)}</time>
                  <span>{event.title}{event.role ? ` · ${event.role}` : ""}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="demo-run-event" aria-live="polite" aria-atomic="true">
            <h4>{active?.title ?? "Before the run"}</h4>
            <p>{active?.detail ?? "Replay the saved run or choose an event to inspect it."}</p>
            {active?.source_at_s !== null && active?.source_at_s !== undefined && (
              <button type="button" onClick={() => seek(active.time_s, active.source_at_s)}>View supporting frame</button>
            )}
          </div>
        </div>
      </div>
      <details className="demo-run-evidence">
        <summary>Available visual evidence ({evidence.length})</summary>
        <ul>{evidence.map((item) => (
          <li key={item.id}><time>{seconds(item.at_s)}</time> <strong>{item.action?.type.replaceAll("_", " ") ?? "Observation"}</strong>
            {item.action?.certainty === "uncertain" ? " (uncertain)" : ""}: {item.description}
            {item.actor?.resolved_name && <small className="demo-run-identity">
              Kit identity: {item.actor.shirt_number !== null ? `#${item.actor.shirt_number}` : item.actor.shirt_name}
              {" → "}{item.actor.resolved_name} · {item.actor.action_role.replaceAll("_", " ")}
            </small>}
          </li>
        ))}</ul>
        {run.inspection && evidence.length > 0 && <p>The observer inspected one region at {[run.inspection.frame_at_s, ...run.inspection.context_at_s].sort((a, b) => a - b).map(seconds).join(", ")} of source time.</p>}
      </details>
      <p className="demo-run-limit"><strong>What to notice:</strong> {run.limitation}</p>
      {run.provenance.reuse_note && <p className="demo-run-caption">{run.provenance.reuse_note}</p>}
      <p className="demo-run-caption">Unedited saved output. Text only; speech durations are simulated. Research was not used in this example. Browsing this run makes no model calls.</p>
    </>
  );
}

export default function AgentRuns({ runs }: { runs: AgentRun[] }) {
  const [activeId, setActiveId] = useState(runs[0].id);
  const active = runs.find((run) => run.id === activeId)!;
  return (
    <section className="demo-agent-runs" id="agent-runs" aria-labelledby="agent-runs-title">
      <h2 id="agent-runs-title">Inside the agents</h2>
      <p className="demo-section-intro">Inspect actual saved runs from the multi-agent prototype. These are separate from the edited, voiced showcase above.</p>
      <label className="demo-picker"><span className="demo-sr-only">Select an automatic agent run</span>
        <select value={activeId} onChange={(event) => setActiveId(event.target.value)}>
          {runs.map((run) => <option value={run.id} key={run.id}>{run.title}</option>)}
        </select>
      </label>
      <RunReview key={active.id} run={active} />
    </section>
  );
}
