"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type savedComparison from "@/data/search-comparison.json";
import "./comparison.css";

export type ComparisonData = typeof savedComparison;
type Hit = { rank: number; title: string; url: string; excerpt: string };
type Search = { provider: string; elapsed_ms: number; results: Hit[]; raw: unknown; request: unknown };
type Fact = { provider: string; rank: number; title: string; url: string; text: string; id: string };
type Packet = { type: string; at_ms: number; provider?: string; query?: string; search?: Search; fact?: Fact; input?: { event: string; fact: Fact }; text?: string; audio_base64?: string; message?: string; run_id?: string };
type SavedRun = { events: Packet[]; query: string; speech_start_s: number; speech_end_s: number; search_source?: { mode: string; run_id: string } | null };
const trigger = 4;
const displayName = (name: string) => name === "keenable" ? "Keenable" : "Exa";

const eventText = "Dan Ballard scores for Sunderland against Arsenal.";

function EventWords({ text }: { text: string }) {
  return text.split(/(Dan Ballard|Sunderland|Arsenal)/g).map((part, i) =>
    /^(Dan Ballard|Sunderland|Arsenal)$/.test(part) ? <mark className="event-word" key={i}>{part}</mark> : part);
}

// Match shared phrases, allowing possessive punctuation. Displayed text stays untouched.
function EvidenceWords({ text, source }: { text: string; source?: string }) {
  if (!source) return <>{text}</>;
  const tokens = /[\p{L}\p{N}]+(?:[’']s)?/gu;
  const normalize = (word: string) => word.toLowerCase().replace(/[’']s$/, "");
  const words = [...text.matchAll(tokens)];
  const sourceWords = (source.match(tokens) || []).map(normalize);
  const matched = new Set<number>();
  for (let i = 0; i < words.length; i++) {
    for (let j = 0; j < sourceWords.length; j++) {
      let n = 0;
      while (i + n < words.length && j + n < sourceWords.length && normalize(words[i + n][0]) === sourceWords[j + n]) n++;
      if (n >= 2) for (let k = i; k < i + n; k++) matched.add(k);
    }
  }
  let end = 0;
  return <>{words.map((word, i) => {
    const start = word.index!;
    const gap = text.slice(end, start); end = start + word[0].length;
    return <span key={i}>{gap}{matched.has(i) ? <mark className="evidence-word">{word[0]}</mark> : word[0]}</span>;
  })}{text.slice(end)}</>;
}

function SearchResponse({ name, result, pending, elapsed, selected, first }: { name: string; result?: Search; pending: boolean; elapsed: number; selected?: Fact; first: boolean }) {
  const picked = selected?.provider === name ? selected : undefined;
  const previews = result?.results.slice(0, 2) || [];
  if (picked && !previews.some(h => h.rank === picked.rank)) {
    const hit = result?.results.find(h => h.rank === picked.rank);
    if (hit) previews.splice(1, 1, hit);
  }
  return <section className="search-response" data-provider={name} data-status={result ? "fulfilled" : pending ? "pending" : "idle"} data-selected={Boolean(picked)}>
    <header><h2>{displayName(name)}{first && <small>first</small>}</h2><span className="response-time">{pending && !result && <i className="search-spinner" />}{result ? `${Math.round(result.elapsed_ms).toLocaleString()} ms` : pending ? `${Math.round(elapsed).toLocaleString()} ms…` : "—"}</span></header>
    {result ? <div className="actual-results">{previews.map(hit => <article className={picked?.rank === hit.rank ? "source-picked" : ""} key={hit.rank}>
      <a href={hit.url} target="_blank" rel="noreferrer">[{hit.rank}] {hit.title}</a>
      <p>{picked?.rank === hit.rank ? <EvidenceWords text={picked.text} source={selected?.text} /> : <>{hit.excerpt.slice(0, 155)}{hit.excerpt.length > 155 ? "…" : ""}</>}</p>
    </article>)}<details><summary>{result.results.length} results</summary>{result.results.map(hit => <article key={hit.rank}><a href={hit.url} target="_blank" rel="noreferrer">[{hit.rank}] {hit.title}</a><pre>{hit.excerpt}</pre></article>)}</details></div> : <div className="response-placeholder" aria-hidden="true"><i /><i /></div>}
  </section>;
}

export default function Comparison({ data, mode = "live", shareable = false }: { data: ComparisonData; mode?: "live" | "replay"; shareable?: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const speech = useRef<HTMLAudioElement | null>(null);
  const request = useRef<AbortController | null>(null);
  const started = useRef(false);
  const startedAt = useRef(0);
  const dispatched = useRef(false);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [liveEvents, setLiveEvents] = useState<Packet[]>([]);
  const [saved, setSaved] = useState<SavedRun | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [provider, setProvider] = useState("keenable");
  const events = useMemo(() => mode === "replay" ? (time >= trigger ? saved?.events.filter(e => e.at_ms <= (time - trigger) * 1000) ?? [] : []) : liveEvents, [mode, time, saved, liveEvents]);
  const returns = events.filter(e => e.type === "search_returned");
  const fact = events.find(e => e.type === "fact_selected")?.fact;
  const draft = events.find(e => e.type === "commentary_ready")?.text;
  const writing = events.some(e => e.type === "commentary_started") && !draft;
  const selecting = events.some(e => e.type === "research_started") && !fact;
  const input = events.find(e => e.type === "commentary_started")?.input;
  const query = events.find(e => e.query)?.query || data.query;
  const flowClock = mode === "replay" ? Math.max(0, time - trigger) * 1000 : elapsed;
  const complete = liveEvents.some(e => e.type === "complete");
  const voicePlaying = mode === "replay" ? Boolean(saved && time >= saved.speech_start_s && time < saved.speech_end_s) : speaking;

  const finishSpeech = useCallback(() => setSpeaking(false), []);

  const applyPacket = useCallback((packet: Packet) => {
    if (packet.type === "search_started" && !dispatched.current) { dispatched.current = true; startedAt.current = performance.now() - packet.at_ms; }
    setLiveEvents(previous => [...previous, packet]);
    if (packet.type === "pipeline_error" || packet.type === "search_error") setError(packet.message || "Research failed.");
    if (packet.type === "audio_ready" && packet.audio_base64) {
      const audio = new Audio(`data:audio/mpeg;base64,${packet.audio_base64}`);
      speech.current = audio;
      audio.onended = finishSpeech;
      audio.onerror = () => { finishSpeech(); setError("Could not play the generated commentary."); };
      audio.onplay = () => {
        const player = video.current;
        if (player) { audio.volume = player.volume; audio.muted = player.muted; }
        setSpeaking(true);
      };
      if (video.current && !video.current.paused) void audio.play().catch(() => setError("Press the video's play control to hear the commentary."));
    }
  }, [finishSpeech]);

  const startRun = useCallback(async () => {
    if (started.current || mode !== "live") return;
    started.current = true; startedAt.current = performance.now();
    request.current = new AbortController();
    try {
      const response = await fetch("/api/search-commentary", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ spend: true }), signal: request.current.signal });
      if (!response.ok) { const body = await response.json(); throw new Error(body.error || "Could not start research."); }
      const reader = response.body?.getReader(); if (!reader) throw new Error("Research stream unavailable.");
      const decoder = new TextDecoder(); let pending = "";
      for (;;) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split("\n"); pending = lines.pop() || "";
        for (const line of lines) if (line.trim()) applyPacket(JSON.parse(line));
        if (done) break;
      }
    } catch (cause) { if (!request.current?.signal.aborted) setError(cause instanceof Error ? cause.message : "Research stopped."); }
  }, [mode, applyPacket]);

  useEffect(() => {
    if (mode === "replay") void fetch("/soccer-commentary/live-search-run.json").then(r => { if (!r.ok) throw new Error("No recorded live run is available."); return r.json(); }).then(setSaved).catch(e => setError(e.message));
    const player = video.current;
    const ready = () => { const at = Number(new URLSearchParams(window.location.search).get("t")); if (player && at > 0 && at <= 20) { player.currentTime = Math.min(at, 19.95); setTime(at); } };
    if (player?.readyState && player.readyState >= 1) ready(); else player?.addEventListener("loadedmetadata", ready, { once: true });
    return () => { request.current?.abort(); player?.removeEventListener("loadedmetadata", ready); player?.pause(); speech.current?.pause(); };
  }, [mode]);

  useEffect(() => {
    if (!playing) return;
    let frame: number;
    const tick = () => { const player = video.current; if (!player) return; setTime(player.currentTime); if (player.currentTime >= trigger) void startRun(); if (!player.ended) frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick); return () => cancelAnimationFrame(frame);
  }, [playing, startRun]);

  useEffect(() => {
    if (mode !== "live" || !liveEvents.length || complete) return;
    const timer = window.setInterval(() => setElapsed(performance.now() - startedAt.current), 50);
    return () => clearInterval(timer);
  }, [mode, liveEvents.length, complete]);

  return <div className="search-demo">
    <main className="outreach-stage simple-replay" data-time={time.toFixed(3)}>
      <header className="simple-header"><span>{shareable ? "commentary" : <a href="/demo">commentary</a>} / {mode === "live" ? "live" : saved?.search_source ? "saved searches" : "replay"}</span></header>
      <div className="simple-workspace">
        <section className="simple-match" aria-label="Match footage and reviewed event">
          <h2 className="step-label">Sunderland v Arsenal</h2>
          <div className="simple-video"><video ref={video} src={mode === "live" ? "/soccer-commentary/ballard-original.mp4?audio=solo" : "/soccer-commentary/ballard-live-replay.mp4?audio=solo"} controls playsInline preload="auto"
            onPlay={() => { setPlaying(true); if (speech.current && !speech.current.ended) void speech.current.play().catch(() => {}); }}
            onPause={() => { setPlaying(false); speech.current?.pause(); }} onEnded={() => { setPlaying(false); setTime(20); }}
            onSeeked={e => setTime(e.currentTarget.currentTime)} onTimeUpdate={e => { if (!playing) setTime(e.currentTarget.currentTime); }}
            onVolumeChange={e => { if (speech.current) { speech.current.volume = e.currentTarget.volume; speech.current.muted = e.currentTarget.muted; } }}
            onError={() => setError("Match video unavailable.")} aria-label="Match footage with generated commentary" /></div>
          <div className="event-input" data-active={time >= trigger}>
            <header>Reviewed event</header>
            {time >= trigger ? <p><EventWords text={eventText} /></p> : <div className="event-waiting" aria-label="Waiting for goal"><i /><i /></div>}
          </div>
        </section>
        <section className="simple-search" aria-label="Shared search query and provider responses">
          <h2 className="step-label">Search</h2>
          <div className="simple-query" data-active={events.length > 0}>
            <header>Query</header>
            <code>{events.length ? <EventWords text={query} /> : <span className="query-waiting">—</span>}</code>
          </div>
          <div className="provider-branches">{["keenable", "exa"].map(name => <SearchResponse key={name} name={name} result={returns.find(e => e.provider === name)?.search} pending={events.some(e => e.type === "search_started" && e.provider === name)} elapsed={flowClock} selected={fact} first={returns[0]?.provider === name} />)}</div>
        </section>
        <section className="simple-commentary" aria-label="Commentary agent">
          <h2 className="step-label">Commentary</h2>
          <div className="commentator-input">
            <header>Input</header>
            <div className="commentary-event" data-active={Boolean(input)}>{input ? <EventWords text={input.event} /> : <span className="quiet">—</span>}</div>
            <section className="fact-handoff" data-active={Boolean(fact)} aria-label="Exact source passage sent to commentary agent">
              <header>{fact ? `${displayName(fact.provider)} [${fact.rank}]` : <>Source{selecting && <i className="search-spinner" role="img" aria-label="Selecting source" />}</>}</header>
              {fact ? <blockquote><a href={fact.url} target="_blank" rel="noreferrer"><EvidenceWords text={fact.text} source={draft || fact.text} /></a></blockquote> : <div className="response-placeholder" aria-hidden="true"><i /><i /></div>}
            </section>
          </div>
          <section className="analyst-output" data-active={Boolean(draft)} aria-label="Exact generated commentary">
            <header><span>Output</span>{voicePlaying ? <i className="voice-indicator" role="img" aria-label="Speaking" /> : writing ? <i className="search-spinner" role="img" aria-label="Writing commentary" /> : events.some(e => e.type === "voice_started") && !events.some(e => e.type === "audio_ready") ? <i className="search-spinner" role="img" aria-label="Preparing audio" /> : null}</header>
            <p>{draft ? <EvidenceWords text={draft} source={fact?.text} /> : <span className="quiet">{writing ? "…" : "—"}</span>}</p>
          </section>
        </section>
      </div>

      {error && <p className="simple-error" role="alert">{error}</p>}
    </main>
    <details className="simple-inspector"><summary>Raw data</summary><div className="simple-inspector-controls"><select value={provider} onChange={e => setProvider(e.target.value)} aria-label="Inspect provider"><option value="keenable">Keenable</option><option value="exa">Exa</option></select>{!shareable && <><a href="/demo/replay?mode=replay">recorded run</a><a href="/demo/replay">live run</a></>}<a href="/soccer-commentary/live-search-run.json" download>trace.json</a></div>{!shareable && <p>Two search calls, up to $0.05 across two model calls, one voice call capped at 240 characters. No retries. Three live runs per server session.</p>}<pre>{JSON.stringify(returns.find(e => e.provider === provider)?.search?.raw ?? {}, null, 2)}</pre><pre>{JSON.stringify({ commentary_input: input, events: events.map(e => e.audio_base64 ? { ...e, audio_base64: "(audio attached)" } : e) }, null, 2)}</pre></details>
  </div>;
}
