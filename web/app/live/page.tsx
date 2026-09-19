"use client";

import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";

type Role = "caller" | "analyst";
type Line = { voice: Role; text: string };
type Frame = { at_s: number; jpeg: string };
type Session = { session_id: string; limits: { duration_s: number; max_batches: number; max_cost_usd: number } };

const MAX_FRAMES = 4;

export default function LivePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const sessionRef = useRef<string | null>(null);
  const generationRef = useRef(0);
  const pickerGenerationRef = useRef(0);
  const framesRef = useRef<Frame[]>([]);
  const startedAtRef = useRef(0);
  const lastDispatchRef = useRef(0);
  const inflightRef = useRef(false);
  const batchesRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const [picking, setPicking] = useState(false);
  const [source, setSource] = useState<"none" | "screen" | "file">("none");
  const [context, setContext] = useState<"generic" | "argfra">("generic");
  const [permission, setPermission] = useState(false);
  const [state, setState] = useState<"ready" | "starting" | "live" | "stopped" | "error">("ready");
  const [message, setMessage] = useState("Choose a screen or local clip to preview it before starting.");
  const [lines, setLines] = useState<Line[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [lag, setLag] = useState<number | null>(null);
  const [cost, setCost] = useState(0);
  const [dropped, setDropped] = useState(0);
  const sourceRef = useRef(source);

  const active = state === "starting" || state === "live";
  useEffect(() => {
    sourceRef.current = source;
  }, [source]);
  const clearSource = useCallback((clearFile = true) => {
    videoRef.current?.pause();
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      if (clearFile) videoRef.current.removeAttribute("src");
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (clearFile && blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
  }, []);
  const stop = useCallback((reason = "Commentary stopped.") => {
    const sessionId = sessionRef.current;
    generationRef.current += 1;
    pickerGenerationRef.current += 1;
    sessionRef.current = null;
    abortRef.current?.abort();
    abortRef.current = null;
    inflightRef.current = false;
    setPicking(false);
    if (sourceRef.current === "screen") {
      clearSource();
      setSource("none");
    } else {
      videoRef.current?.pause();
    }
    setState("stopped");
    setMessage(reason);
    if (sessionId) void fetch("/api/live", { method: "POST", keepalive: true, headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "stop", session_id: sessionId }) }).catch(() => undefined);
  }, [clearSource]);

  useEffect(() => () => {
    generationRef.current += 1;
    pickerGenerationRef.current += 1;
    abortRef.current?.abort();
    const sessionId = sessionRef.current;
    sessionRef.current = null;
    clearSource();
    if (sessionId) void fetch("/api/live", { method: "POST", keepalive: true, headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "stop", session_id: sessionId }) }).catch(() => undefined);
  }, [clearSource]);
  useEffect(() => {
    if (state !== "live") return;
    const timer = window.setInterval(() => setElapsed((performance.now() - startedAtRef.current) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [state]);

  const shareScreen = async () => {
    const pickerGeneration = ++pickerGenerationRef.current;
    setPicking(true);
    setSource("none");
    try {
      clearSource();
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      if (pickerGeneration !== pickerGenerationRef.current || active) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      stream.getVideoTracks()[0]?.addEventListener("ended", () => stop("Screen sharing ended."));
      const video = videoRef.current!;
      video.removeAttribute("src");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      setSource("screen");
      setMessage("Screen preview ready. Review it, then explicitly start paid commentary.");
    } catch (error) {
      if (pickerGeneration !== pickerGenerationRef.current) return;
      setState("error");
      setMessage(error instanceof Error ? `Could not share this screen: ${error.message}` : "Screen sharing was not started.");
      clearSource();
    } finally {
      if (pickerGeneration === pickerGenerationRef.current) setPicking(false);
    }
  };
  const chooseClip = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !videoRef.current) return;
    pickerGenerationRef.current += 1;
    clearSource();
    const blobUrl = URL.createObjectURL(file);
    blobUrlRef.current = blobUrl;
    videoRef.current.src = blobUrl;
    videoRef.current.muted = true;
    setSource("file");
    setMessage("Local clip selected. Press play to preview; commentary samples only while it plays.");
  };

  const capture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!sessionRef.current || !video || !canvas || video.paused || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    const scale = Math.min(1, 1280 / (video.videoWidth || 1280), 720 / (video.videoHeight || 720));
    const width = Math.round((video.videoWidth || 1280) * scale);
    const height = Math.round((video.videoHeight || 720) * scale);
    let jpeg: string | undefined;
    try {
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);
      jpeg = canvas.toDataURL("image/jpeg", 0.82).split(",")[1];
    } catch {
      stop("Frame capture failed; commentary stopped.");
      return;
    }
    if (!jpeg) return;
    const next = { at_s: Number(((performance.now() - startedAtRef.current) / 1000).toFixed(2)), jpeg };
    if (framesRef.current.length === MAX_FRAMES) setDropped((value) => value + 1);
    framesRef.current = [...framesRef.current.slice(-(MAX_FRAMES - 1)), next];
  }, [stop]);
  const sendBatch = useCallback(async (generation: number) => {
    if (inflightRef.current || !sessionRef.current || batchesRef.current >= 12) return;
    if (performance.now() - lastDispatchRef.current < 4000) return;
    const frames = framesRef.current;
    if (!frames.length) return;
    inflightRef.current = true;
    framesRef.current = [];
    const requestStartedAt = performance.now();
    lastDispatchRef.current = requestStartedAt;
    const playback_s = Number(((requestStartedAt - startedAtRef.current) / 1000).toFixed(2));
    if (playback_s >= 60) { stop("The 60-second session limit was reached."); return; }
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch("/api/live", { method: "POST", signal: controller.signal, headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "batch", session_id: sessionRef.current, playback_s, frames }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "The commentary service rejected this batch.");
      if (generation !== generationRef.current) return;
      batchesRef.current += 1;
      if (body.line?.text) setLines((value) => [...value, body.line]);
      setCost(body.usage?.cumulative?.cost_usd ?? 0);
      setLag((performance.now() - requestStartedAt) / 1000);
      if (body.stopped_reason) stop(body.stopped_reason);
      if (batchesRef.current >= 12) stop("The 12-batch session limit was reached.");
    } catch (error) {
      if (generation === generationRef.current) stop(error instanceof Error ? error.message : "Commentary stopped after a server error.");
    } finally {
      if (generation === generationRef.current) { inflightRef.current = false; abortRef.current = null; }
    }
  }, [stop]);
  useEffect(() => {
    if (state !== "live") return;
    const generation = generationRef.current;
    const captureTimer = window.setInterval(capture, 1000);
    const batchTimer = window.setInterval(() => void sendBatch(generation), 250);
    const limitTimer = window.setTimeout(() => stop("The 60-second session limit was reached."), 60000);
    return () => { window.clearInterval(captureTimer); window.clearInterval(batchTimer); window.clearTimeout(limitTimer); };
  }, [capture, sendBatch, state, stop]);

  const start = async () => {
    if (!permission || source === "none") { setMessage("Select a preview source and confirm the paid-service notice before starting."); return; }
    setState("starting"); setMessage("Starting the paid, time-limited session…"); setLines([]); setCost(0); setDropped(0); framesRef.current = []; batchesRef.current = 0;
    const generation = ++generationRef.current;
    inflightRef.current = false;
    setElapsed(0);
    setLag(null);
    try {
      const response = await fetch("/api/live", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "start", context, spend: true }) });
      const body: Session | { error: string } = await response.json();
      if (!response.ok || "error" in body) throw new Error("error" in body ? body.error : "Could not start commentary.");
      if (generation !== generationRef.current) {
        void fetch("/api/live", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "stop", session_id: body.session_id }) }).catch(() => undefined);
        return;
      }
      sessionRef.current = body.session_id;
      startedAtRef.current = performance.now();
      lastDispatchRef.current = startedAtRef.current;
      if (source === "file") {
        try {
          await videoRef.current?.play();
        } catch (error) {
          stop(error instanceof Error ? `Local video could not play: ${error.message}` : "Local video could not play.");
          return;
        }
      }
      if (generation !== generationRef.current) return;
      setState("live");
      setMessage("Live analysis is running. Video remains muted; no speech is played.");
    } catch (error) {
      if (generation === generationRef.current) {
        const sessionId = sessionRef.current;
        sessionRef.current = null;
        if (sessionId) {
          void fetch("/api/live", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "stop", session_id: sessionId }) }).catch(() => undefined);
        }
        setState("error");
        setMessage(error instanceof Error ? error.message : "Could not start commentary.");
      }
    }
  };

  return (
    <main className="min-h-screen px-4 py-6 sm:px-6 sm:py-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-6 flex flex-wrap items-baseline justify-between gap-3 border-b border-line pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Live screen commentary</h1>
            <p className="mt-1 text-sm text-dim">A muted screen-streaming test. No audio is captured or played.</p>
          </div>
          <a href="/demo" className="text-sm text-dim underline decoration-line underline-offset-4 hover:text-text focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller">Saved demo</a>
        </header>
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(19rem,0.8fr)]">
          <section>
            <div className="aspect-video overflow-hidden border border-line bg-panel">
              <video ref={videoRef} controls playsInline muted className="h-full w-full object-contain" aria-label="Selected match screen preview" onEnded={() => { if (active) stop("The local clip ended."); }} onError={() => setMessage("The selected video could not be played.")} />
            </div>
            <canvas ref={canvasRef} className="hidden" />
            <div className="mt-3 flex flex-wrap gap-2">
              <button onClick={() => void shareScreen()} disabled={active || picking} className="rounded bg-caller px-4 py-2 text-sm font-medium text-ink disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller">Share match screen</button>
              <label className="cursor-pointer rounded border border-line px-4 py-2 text-sm text-text hover:border-dim focus-within:outline-2 focus-within:outline-offset-4 focus-within:outline-caller">
                Choose local clip
                <input type="file" accept="video/*" className="sr-only" onChange={chooseClip} disabled={active || picking} />
              </label>
            </div>
            <p className="mt-3 text-sm leading-6 text-dim">Share only the tab or window you intend to analyze. Sampled frames are sent to OpenAI; no audio is sent and nothing is saved locally.</p>
          </section>
          <aside className="border border-line bg-panel p-4">
            <div className="flex items-center justify-between"><h2 className="font-medium">Commentary</h2><span className="text-sm text-dim">{state}</span></div>
            <p className="mt-2 text-sm text-dim">Caller: Terra · Analyst: Terra</p>
            <div className="mt-3 space-y-1 text-sm text-dim"><p>Elapsed: {elapsed.toFixed(1)}s / 60s</p><p>Request latency: {lag === null ? "waiting" : `${lag.toFixed(2)}s`}</p><p>Cost: ${cost.toFixed(3)}</p><p>Unsent samples skipped: {dropped}</p></div>
            <div className="mt-4 min-h-48 space-y-3 border-y border-line py-4 text-sm" aria-live="polite">{lines.length ? lines.map((line, index) => <p key={`${index}-${line.text}`}><span className={line.voice === "caller" ? "text-caller" : "text-analyst"}>{line.voice === "caller" ? "Caller" : "Analyst"}</span><span className="sr-only">: </span><span className="text-text"> — {line.text}</span></p>) : <p className="text-dim">No commentary yet. Captured frames are sent every four seconds once the session begins.</p>}</div>
            <p className={state === "error" ? "mt-3 text-sm text-analyst" : "mt-3 text-sm text-dim"}>{message}</p>
            <fieldset className="mt-5 border-t border-line pt-4" disabled={active || picking}><label className="block text-sm" htmlFor="context">Match context</label><select id="context" value={context} onChange={(event) => setContext(event.target.value as "generic" | "argfra")} className="mt-1 w-full rounded border border-line bg-ink px-3 py-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-caller"><option value="generic">Generic match</option><option value="argfra">Argentina–France</option></select><label className="mt-4 flex gap-2 text-sm leading-5"><input type="checkbox" checked={permission} onChange={(event) => setPermission(event.target.checked)} className="mt-1 focus-visible:outline-2 focus-visible:outline-caller" />I understand sampled frames are sent to a paid service. $0.35 is an estimated token stop; the last request may exceed it.</label></fieldset>
            <div className="mt-4 flex gap-2"><button onClick={() => void start()} disabled={active || picking || source === "none" || !permission} className="rounded bg-text px-4 py-2 text-sm font-medium text-ink disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller">Start commentary</button>{active && <button onClick={() => stop()} className="rounded border border-line px-4 py-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller">Stop</button>}</div>
          </aside>
        </div>
      </div>
    </main>
  );
}
