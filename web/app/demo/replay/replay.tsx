"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type recordedTrace from "@/public/soccer-commentary/prepared-penalty.trace.json";
import "./replay.css";

type Trace = typeof recordedTrace;
type Delivery = { voice: string; start: number; end: number; text: string };
const seconds = (n: number) => `${n.toFixed(3)}s`;

export default function Replay({ trace, delivery }: { trace: Trace; delivery: Delivery[] }) {
  const router = useRouter();
  const video = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const [speed, setSpeed] = useState(1);
  const [muted, setMuted] = useState(false);
  const [pinned, setPinned] = useState<number | "identity" | null>(null);
  const [filter, setFilter] = useState("");
  const [tab, setTab] = useState<"turn" | "observations" | "config">("turn");
  const turns = trace.dialogue.turns;
  const autoIndex = turns.findLastIndex(t => t.source_at_s <= time + 0.005);
  const identity = pinned === "identity" || (pinned === null && time >= 9 && time < 10);
  const turnIndex = typeof pinned === "number" ? pinned : autoIndex;
  const turn = turns[turnIndex];
  const line = delivery.find(l => time >= l.start && time < l.end);
  const rejection = turn && "rejected" in turn ? turn.rejected : undefined;
  const observationIds = identity ? ["recorded-3", "recorded-4"] : turn?.result.line?.evidence_ids ?? (time >= 17 ? ["recorded-6"] : turn?.visible_evidence.slice(-1) ?? []);
  // The identity batch spans 9–12s; only show frames at the selected match cue.
  const observations = trace.observations.filter(o => observationIds.includes(o.id) && (!identity || o.at_s <= time));
  const state = identity ? "research_roster / identify" : rejection ? "rejected" : turn?.result.line ? "speak" : turn ? "abstain" : "source";
  const raw = tab === "config" ? { source: trace.source, model: trace.model, budget: trace.budget, vision_reuse: trace.vision_reuse, observer_system: trace.system }
    : tab === "observations" ? observations
    : identity ? trace.identity_steps.slice(0, 2) : turn ?? { source: trace.source, vision_reuse: trace.vision_reuse };
  useEffect(() => { const player = video.current; return () => player?.pause(); }, []);

  function seek(at: number, selection: number | "identity" | null = null) {
    if (!video.current) return;
    video.current.pause(); video.current.currentTime = at; setTime(at); setPinned(selection);
  }
  async function play(restart = false) {
    const player = video.current; if (!player) return;
    if (restart || player.ended) { player.currentTime = 0; setTime(0); }
    setPinned(null);
    try { await player.play(); } catch { setError("Playback failed. Try the native video controls or reload."); }
  }
  return <main className="replay-page">
    <header className="dev-header"><div><a href="/demo">commentary</a><span>/</span><h1>run inspector</h1><select aria-label="Choose recorded run" value="penalty" onChange={e => router.push(`/demo/replay?run=${e.target.value}`)}><option value="search">ballard / search comparison</option><option value="penalty">penalty / agent turns</option></select></div><div><span className="dev-mode">offline replay</span><a href="/soccer-commentary/prepared-penalty.trace.json" download>download trace.json</a><button onClick={() => play(true)}>restart</button></div></header>
    <div className="dev-run-meta"><span>source: <b>england-france-penalties.mp4 [42–62s]</b></span><span>observer: <b>{trace.model}</b></span><span>dialogue: <b>gpt-5.4-mini</b></span><span>cost: <b>${trace.budget.reported_usd.toFixed(5)}*</b></span></div>
    <div className="replay-workspace">
      <div className="dev-left">
        <section className="replay-match" aria-label="Match footage and audio">
          <div className="dev-pane-title"><span>prepared-penalty-agents.mp4</span><span>20.000s · synthesized caller + analyst</span></div>
          <div className="replay-video"><video ref={video} src="/soccer-commentary/prepared-penalty-agents.mp4" playsInline preload="auto" controls muted={muted}
            onLoadedMetadata={e => { e.currentTarget.playbackRate = speed; }}
            onTimeUpdate={e => setTime(e.currentTarget.currentTime)} onSeeked={e => setTime(e.currentTarget.currentTime)}
            onPlay={() => { setPlaying(true); setPinned(null); setError(""); }} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
            onVolumeChange={e => setMuted(e.currentTarget.muted)}
            onError={() => setError("Missing media: web/public/soccer-commentary/prepared-penalty-agents.mp4. Run node scripts/prepare_replay.mjs after restoring source media.")}
            aria-label="England France penalty with original AI caller and analyst audio" /></div>
          {error && <p className="replay-error" role="alert">{error}</p>}
          <div className="replay-transport"><button aria-label={playing ? "Pause playback" : "Resume playback"} onClick={() => playing ? video.current?.pause() : play()}>{playing ? "pause" : "play"}</button><span>{seconds(time)}</span><input type="range" min="0" max="20" step="0.01" value={time} aria-label="Match timeline" onChange={e => seek(Number(e.target.value))} /><button onClick={() => setMuted(!muted)} aria-pressed={muted}>{muted ? "unmute" : "mute"}</button><select aria-label="Playback speed" value={speed} onChange={e => { const s = Number(e.target.value); setSpeed(s); if (video.current) video.current.playbackRate = s; }}><option value="1">1×</option><option value="0.75">0.75×</option><option value="0.5">0.5×</option></select></div>
          <div className="replay-caption" data-voice={line?.voice}><span>audio/{line?.voice ?? "idle"}</span><p>{line?.text ?? "—"}</p></div>
        </section>
        <section className="dev-trace" aria-label="Agent turns">
          <div className="dev-pane-title"><span>dialogue.turns [{turns.length}]</span><label><input type="checkbox" checked={pinned === null} onChange={e => setPinned(e.target.checked ? null : identity ? "identity" : turnIndex)} />follow playback</label></div>
          <div className="dev-trace-tools"><input type="search" aria-label="Filter turns" placeholder="filter role, action, text…" value={filter} onChange={e => setFilter(e.target.value)} /><button className={identity ? "dev-selected-tool" : ""} onClick={() => seek(9, "identity")}>identity_steps [4]</button></div>
          <div className="dev-table-scroll"><table><thead><tr><th>clip t</th><th>role</th><th>action</th><th>output</th><th>latency</th></tr></thead><tbody>{turns.map((t, i) => {
            const rejected = "rejected" in t ? t.rejected : undefined;
            const action = rejected ? "rejected" : t.result.line ? "speak" : "abstain";
            if (!`${t.role} ${action} ${t.result.line?.text ?? t.result.reason}`.toLowerCase().includes(filter.toLowerCase())) return null;
            return <tr key={i} className={!identity && turnIndex === i ? "dev-selected" : ""} data-future={t.source_at_s > time + .005}>
              <td><button aria-label={`Inspect ${t.role} ${action} at ${seconds(t.source_at_s)}`} onClick={() => seek(t.source_at_s, i)}>{t.source_at_s.toFixed(3)}</button></td><td>{t.role}</td><td className={rejected ? "dev-rejection" : ""}>{action}</td><td><button onClick={() => seek(t.source_at_s, i)} title={t.result.line?.text ?? t.result.reason}>{t.result.line?.text ?? t.result.reason}</button></td><td>{t.result.trace[0].data.usage.latency_s.toFixed(2)}s</td>
            </tr>;
          })}</tbody></table></div>
          <div className="dev-trace-note">clip t = source cue · latency = model request time · click a row to pause + inspect</div>
        </section>
      </div>
      <section className="replay-inspector" aria-label="Evidence and agent decisions">
        <div className="dev-pane-title"><span>{identity ? "identity_steps[0:2]" : turn ? `dialogue.turns[${turnIndex}]` : "result.json"}</span><span className={rejection && !identity ? "dev-rejection" : ""}>{state}</span></div>
        <div className="dev-selection-meta">{identity ? <span>batch: recorded-3, recorded-4, recorded-5 · grouped at 9s; no execution timestamp</span> : turn ? <><span>cutoff={seconds(turn.result.cutoff_s)}</span><span>words≤{turn.word_budget}</span><span>generation_elapsed={seconds(turn.generation_elapsed_s)}</span></> : <span>select a turn or play the saved render</span>}</div>
        <div className="replay-evidence">
          {observations.map(o => <div key={o.id} className="replay-observation">
            <div className="dev-observation-title"><code>{o.id}</code><span>{o.action.type} / {o.action.certainty}</span></div>
            <div className="replay-frames">{o.action.support_at_s.map(at => <figure key={at}><img src={`/soccer-commentary/replay-frames/${at}.jpg`} alt={`Source footage at clip second ${at}, supporting ${o.id}`} /><figcaption>clip={at.toFixed(1)}s source={(at + 42).toFixed(1)}s</figcaption></figure>)}</div>
            <p>{o.description}</p><div className="replay-cues">cues: {o.action.cues.join(", ") || "[]"}</div>
            {o.actor?.resolved_name && <div className="replay-identity-link">actor: #{o.actor.shirt_number} → {o.actor.resolved_name} <button onClick={() => seek(9, "identity")}>[identity_steps]</button></div>}
          </div>)}
          {identity ? <div className="replay-decision"><p><b>research_roster</b> → England #9 → Harry Kane</p><p>{trace.identity_steps[0].action.reason}</p><a href={trace.research.squads[0].source_url} target="_blank" rel="noreferrer">{trace.research.squads[0].id} ({trace.research.squads[0].source_date})</a></div> : turn ? <div className="replay-decision"><p className={rejection ? "dev-rejection" : ""}><b>{rejection ? "validation_error" : "reason"}:</b> {rejection ? String(rejection) : turn.result.reason}</p><p className="dev-available">visible_evidence: [{turn.visible_evidence.join(", ")}]</p></div> : <pre className="dev-source">{JSON.stringify(trace.source, null, 2)}</pre>}
        </div>
        <div className="dev-payload-tabs" aria-label="Payload view">{([['turn','agent payload'],['observations','observations'],['config','run config']] as const).map(([value,label]) => <button key={value} aria-pressed={tab === value} onClick={() => setTab(value)}>{label}</button>)}</div>
        <pre className="replay-raw" tabIndex={0} aria-label="Raw saved record">{JSON.stringify(raw, null, 2)}</pre>
      </section>
    </div>
    <footer className="replay-footer"><span>saved trace: web/public/soccer-commentary/prepared-penalty.trace.json</span><span>* final dialogue only; reused vision + identity excluded</span></footer>
    <details className="replay-provenance"><summary>preparation / timing notes</summary><p>Full-sequence observer: 24s context. Dialogue uses source-time evidence cutoffs. This is prepared commentary, not causal live inference. Identity tool steps are grouped by relevant evidence, not execution time. Agent descriptions are saved model outputs. Playback makes no model calls.</p><p>Render placement: examples/recorded-agent-demo/render-report.md. Models and inherited usage fields describe their originating phases. No commentary text has been edited for this viewer.</p></details>
  </main>;
}
