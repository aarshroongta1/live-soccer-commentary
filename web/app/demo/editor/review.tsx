"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

const subscribeToVoiceSupport = () => () => {};
const getVoiceSupport = () => "speechSynthesis" in window;
const getServerVoiceSupport = () => false;

type Decision = "keep" | "rewrite" | "reject";
type Review = { decision: Decision; reason: string; rewrite: string | null; trajectory_url: string };
type Take = { label: string; text: string; trajectory_url: string };
type Session = { policy?: string; connected?: boolean; review?: Review; takes?: Take[]; comparison_status?: string;
  vote?: { choice: string; reason: string }; reveal?: Record<string, string>; error?: string;
  recorded?: boolean; recorded_order?: Record<string, string>; owner_vote?: {choice: string; reason: string} };
type Data = { title: string; proposed: string; video_url: string; start_s: number; end_s: number;
  source_origin: string; fact: { text: string; url: string }; transfer: { title: string; video_url: string; cutoff_s: number;
    research?: {id: string; text: string; url: string}[]; research_limit?: string } };

export default function Editor({ data }: { data: Data }) {
  const player = useRef<HTMLVideoElement>(null);
  const playbackWindow = useRef(false);
  const requestId = useRef("");
  const [session, setSession] = useState<Session>({});
  const [loaded, setLoaded] = useState(false);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [rewrite, setRewrite] = useState(data.proposed);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [step, setStep] = useState<"review" | "compare">("review");
  const [vote, setVote] = useState("");
  const [voteReason, setVoteReason] = useState("");
  const [speaking, setSpeaking] = useState("");
  const voiceAvailable = useSyncExternalStore(subscribeToVoiceSupport, getVoiceSupport, getServerVoiceSupport);

  useEffect(() => {
    let live = true;
    fetch("/api/commentary-editor").then(r => r.json()).then(async s => {
      if (!s.connected && !s.error) {
        const replay = await fetch("/editor-review/session.json").then(r => r.json());
        if (replay.recorded) s = { ...replay, connected: false };
      }
      if (!live) return;
      if (s.error) throw new Error(s.error);
      setSession(s); setLoaded(true);
      if (s.review) {
        setDecision(s.review.decision); setReason(s.review.reason);
        setRewrite(s.review.rewrite || data.proposed);
        setStatus("");
      } else {
        try {
          const draft = JSON.parse(localStorage.getItem("commentary-editor-draft") || "null");
          if (draft && ["keep", "rewrite", "reject"].includes(draft.decision)) {
            setDecision(draft.decision); setReason(draft.reason || "");
            setRewrite(draft.rewrite || data.proposed);
          }
        } catch { /* An invalid or unavailable local draft cannot affect cloud state. */ }
      }
    }).catch(() => { if (live) { setLoaded(true); setError("Could not connect to the review session."); } });
    return () => { live = false; window.speechSynthesis?.cancel(); };
  }, [data.proposed]);

  useEffect(() => {
    if (!loaded || session.review) return;
    try { localStorage.setItem("commentary-editor-draft", JSON.stringify({ decision, rewrite, reason })); }
    catch { /* Keep the current form usable when storage is unavailable. */ }
  }, [loaded, decision, rewrite, reason, session.review]);

  async function send(body: object) {
    setBusy(true); setError(""); setStatus("");
    try {
      const r = await fetch("/api/commentary-editor", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await r.json();
      if (!r.ok || result.error) throw new Error(result.error || "Could not save.");
      setSession(result);
      return result as Session;
    } catch (e) { setError(e instanceof Error ? e.message : "Could not save."); return null; }
    finally { setBusy(false); }
  }

  function change() { requestId.current = ""; setStatus(""); }
  async function save() {
    if (!decision || !reason.trim()) { setError("Choose a decision and add your reason."); return; }
    if (decision === "rewrite" && (!rewrite.trim() || rewrite.trim() === data.proposed)) {
      setError("Change the line, or choose Keep."); return;
    }
    if (!session.connected) {
      const value = { decision, reason, rewrite: decision === "rewrite" ? rewrite : null };
      try { localStorage.setItem("commentary-editor-preview", JSON.stringify(value));
        setStatus("Saved in this browser.");
      } catch { setError("Browser storage is unavailable."); }
      return;
    }
    if (!requestId.current) requestId.current = crypto.randomUUID();
    const result = await send({ action: "review", decision, reason, rewrite,
      playback_s: player.current?.currentTime || 0, request_id: requestId.current });
    if (result) {
      setStatus("Saved to Trajectory.");
      try { localStorage.removeItem("commentary-editor-draft"); } catch { /* Cloud save is verified. */ }
    }
  }

  function hearLine() {
    window.speechSynthesis?.cancel();
    if (!player.current) return;
    playbackWindow.current = true;
    player.current.currentTime = Math.max(0, data.start_s - 2);
    player.current.muted = false;
    player.current.play().catch(() => setError("Use the video’s play control to start audio."));
  }

  function preview(text: string, label: string) {
    if (!voiceAvailable) return;
    window.speechSynthesis.cancel(); player.current?.pause();
    const utterance = new SpeechSynthesisUtterance(text === "[SILENCE]" ? "No commentary." : text);
    utterance.onend = () => setSpeaking(""); utterance.onerror = () => setSpeaking("");
    setSpeaking(label); window.speechSynthesis.speak(utterance);
  }

  async function generate() {
    const result = await send({ action: "compare" });
    if (result) setStatus("Both takes are ready.");
  }

  async function saveVote() {
    if (session.recorded) {
      const judgment = { choice: vote, reason: voteReason };
      try { localStorage.setItem("commentary-editor-visitor-judgment", JSON.stringify(judgment)); } catch { /* Reading still works without browser storage. */ }
      setSession({ ...session, vote: judgment, reveal: session.recorded_order });
      setStatus("Revealed. Your judgment stays in this browser.");
      return;
    }
    const result = await send({ action: "vote", choice: vote, reason: voteReason });
    if (result) setStatus("Judgment saved to Trajectory.");
  }

  return <main className="editor-shell">
    <header className="editor-header">
      <h1>Commentary review</h1>
      <span>{!loaded ? "Connecting…" : session.connected ? "Connected to Trajectory" : session.recorded ? "Recorded example" : "Preview · saved locally"}</span>
    </header>
    <nav className="editor-tabs" aria-label="Review workflow">
      <button aria-current={step === "review" ? "step" : undefined} onClick={() => { setStep("review"); setError(""); }}>Review</button>
      <button disabled={!session.review} aria-current={step === "compare" ? "step" : undefined} onClick={() => { setStep("compare"); setError(""); }}>Compare</button>
    </nav>
    {step === "review" ? <div className="editor-workspace">
      <section className="editor-footage">
        <h2>{data.title}</h2>
        <video ref={player} src={data.video_url} controls playsInline preload="metadata" aria-label="Sunderland versus Arsenal footage with generated commentary" onTimeUpdate={() => {
          if (playbackWindow.current && player.current && player.current.currentTime >= data.end_s) {
            player.current.pause(); playbackWindow.current = false;
          }
        }} />
        <button className="editor-secondary editor-replay" onClick={hearLine}>Replay the line</button>
        <details><summary>Source</summary><blockquote>{data.fact.text}</blockquote><a href={data.fact.url} target="_blank" rel="noreferrer">Read source</a></details>
      </section>
      <section className="editor-form" aria-labelledby="proposed-title">
        <h2 id="proposed-title">Proposed line</h2>
        <blockquote className="editor-proposal">{data.proposed}</blockquote>
        <div className="editor-decisions" role="group" aria-label="Your decision">
          {(["keep", "rewrite", "reject"] as const).map(action => <button key={action} aria-pressed={decision === action} disabled={!loaded || busy || !!session.comparison_status} onClick={() => { setDecision(action); change(); }}>{action[0].toUpperCase() + action.slice(1)}</button>)}
        </div>
        {decision === "rewrite" && <div className="editor-field">
          <label htmlFor="rewrite">Your edit</label>
          <textarea id="rewrite" maxLength={500} rows={3} value={rewrite} disabled={!loaded || busy || !!session.comparison_status} onChange={e => { setRewrite(e.target.value); change(); }} />
          {voiceAvailable && <button className="editor-text-button" onClick={() => preview(rewrite, "edit")}>{speaking === "edit" ? "Playing…" : "Listen"}</button>}
        </div>}
        <div className="editor-field">
          <label htmlFor="reason">Reason</label>
          <textarea id="reason" rows={2} maxLength={1200} value={reason} disabled={!loaded || busy || !!session.comparison_status} onChange={e => { setReason(e.target.value); change(); }} />
        </div>
        {session.policy && <div className="editor-field"><label>Preference inferred from your review</label><p>{session.policy}</p></div>}
        <div className="editor-actions">
          {!session.recorded && !session.comparison_status && <button className="editor-primary" disabled={!loaded || busy} onClick={save}>{busy ? "Saving…" : "Save review"}</button>}
          {session.review && <a href={session.recorded ? "/editor-review/evidence.json" : session.review.trajectory_url} target="_blank" rel="noreferrer">{session.recorded ? "Evidence" : "View in Trajectory"}</a>}
        </div>
      </section>
    </div> : <div className="editor-workspace">
      <section className="editor-footage">
        <h2>{data.transfer.title}</h2>
        <video src={data.transfer.video_url} muted controls playsInline preload="metadata" aria-label="Barcelona versus Bayern footage for the commentary comparison" />
        <details><summary>Background</summary><ul>{data.transfer.research?.map(fact => <li key={fact.id}>{fact.text} <a href={fact.url} target="_blank" rel="noreferrer">Source</a></li>)}</ul></details>
      </section>
      <section className="editor-comparison" aria-label="Compare commentary">
        {!session.takes ? <>
          <h2>Compare two takes</h2>
          <p>Same footage and background. One take also uses your feedback.</p>
          <button className="editor-primary" disabled={busy || !!session.comparison_status} onClick={generate}>{busy ? "Generating…" : "Generate takes"}</button>
        </> : <>
          <div className="editor-takes">{session.takes.map(take => <article key={take.label}>
            <div className="editor-take-heading"><h2>{take.label}</h2>
              {session.reveal && <span>{session.reveal[take.label] === "with_feedback" ? "With feedback" : "Baseline"}</span>}
              {voiceAvailable && <button className="editor-text-button" onClick={() => preview(take.text, take.label)}>{speaking === take.label ? "Playing…" : "Listen"}</button>}
            </div>
            <blockquote>{take.text === "[SILENCE]" ? "Stay silent." : take.text}</blockquote>
            {session.reveal && <a className="editor-run-link" href={session.recorded ? "/editor-review/evidence.json" : take.trajectory_url} target="_blank" rel="noreferrer">View run</a>}
          </article>)}</div>
          <div className="editor-vote">
            <fieldset><legend>Which would you air?</legend>{["A", "B", "tie", "neither"].map(choice => <label key={choice}>
              <input type="radio" name="take" value={choice} checked={vote === choice} disabled={busy || !!session.vote} onChange={() => setVote(choice)} />
              {choice === "tie" ? "Equal" : choice === "neither" ? "Neither" : choice}
            </label>)}</fieldset>
            <label htmlFor="vote-reason">Reason</label>
            <textarea id="vote-reason" rows={2} maxLength={1200} value={voteReason} disabled={busy || !!session.vote} onChange={e => setVoteReason(e.target.value)} />
            {!session.vote && <button className="editor-primary" disabled={busy || !vote || !voteReason.trim()} onClick={saveVote}>{busy ? "Saving…" : session.reveal ? "Save judgment" : "Save and reveal"}</button>}
            {session.vote && <p className="editor-result">Your choice: {session.vote.choice}. {session.vote.reason}</p>}
            {session.reveal && session.owner_vote && <p className="editor-result">Original review: {session.owner_vote.choice}. {session.owner_vote.reason}</p>}
          </div>
        </>}
      </section>
    </div>}
    <div className="editor-feedback" aria-live="polite">{status && <p>{status}</p>}{error && <p role="alert" className="editor-error">{error}</p>}</div>
    <footer className="editor-footer">
      <details><summary>Details</summary>
        <p>{data.source_origin} The line begins at {data.start_s.toFixed(1)}s. Edit and comparison previews use a browser voice.</p>
        <p>The comparison uses the same model, frames, and background for both takes. The revised take uses a preference inferred from your saved review, plus an instruction to favor one relevant background fact. The first attempt using raw feedback did not add background context; both attempts are in the evidence. This is learning through saved instructions, with no model training. No human preference between the final takes has been recorded.</p>
        {data.transfer.research_limit && <p>{data.transfer.research_limit}</p>}
        {!session.connected && !session.recorded && <p>This preview saves decisions in your browser. Live Trajectory capture is available in the connected local session.</p>}
      </details>
      <a href="https://www.aarshroongta.com/projects/soccer-commentary">Project</a>
    </footer>
  </main>;
}
