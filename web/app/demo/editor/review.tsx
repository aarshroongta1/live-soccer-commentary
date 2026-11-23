"use client";

import { useEffect, useRef, useState } from "react";

type Decision = "keep" | "rewrite" | "reject";
type Review = { decision: Decision; reason: string; rewrite: string | null; trajectory_url: string };
type Take = { label: string; text: string; trajectory_url: string };
type Session = { connected?: boolean; review?: Review; takes?: Take[]; comparison_status?: string;
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
  const [voiceAvailable, setVoiceAvailable] = useState(false);

  useEffect(() => {
    let live = true;
    setVoiceAvailable("speechSynthesis" in window);
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
        setStatus(s.recorded ? "Recorded owner review, verified against Trajectory." : "Saved to Trajectory and verified.");
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
        setStatus("Saved in this browser. Preview decisions are not sent to Trajectory.");
      } catch { setError("Browser storage is unavailable."); }
      return;
    }
    if (!requestId.current) requestId.current = crypto.randomUUID();
    const result = await send({ action: "review", decision, reason, rewrite,
      playback_s: player.current?.currentTime || 0, request_id: requestId.current });
    if (result) {
      setStatus("Saved to Trajectory and verified.");
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
    if (result) setStatus("Both takes recorded. Judge them before revealing which used your feedback.");
  }

  async function saveVote() {
    if (session.recorded) {
      const judgment = { choice: vote, reason: voteReason };
      try { localStorage.setItem("commentary-editor-visitor-judgment", JSON.stringify(judgment)); } catch { /* Reading still works without browser storage. */ }
      setSession({ ...session, vote: judgment, reveal: session.recorded_order });
      setStatus("Your judgment stays in this browser. The recorded runs are now revealed.");
      return;
    }
    const result = await send({ action: "vote", choice: vote, reason: voteReason });
    if (result) setStatus("Your comparison judgment is saved to Trajectory and verified.");
  }

  return <main className="editor-shell">
    <header className="editor-header"><a href="https://www.aarshroongta.com">Aarsh Roongta</a>
      <span>{!loaded ? "Connecting…" : session.connected ? "Trajectory connected" : session.recorded ? "Recorded learning experiment" : "Interactive preview"}</span></header>
    <div className="editor-intro"><h1>Teach the commentary agent.</h1><p>Watch the play. Make the line better. See whether your feedback helps on another clip.</p></div>
    <nav className="editor-tabs" aria-label="Review workflow">
      <button aria-current={step === "review" ? "step" : undefined} onClick={() => { setStep("review"); setError(""); }}>Review a line</button>
      <button disabled={!session.review} aria-current={step === "compare" ? "step" : undefined} onClick={() => { setStep("compare"); setError(""); }}>Test on another clip</button>
    </nav>
    {step === "review" ? <div className="editor-workspace">
      <section className="editor-footage"><div className="editor-section-heading"><h2>{data.title}</h2><span>Recorded footage and AI voices</span></div>
        <video ref={player} src={data.video_url} controls playsInline preload="metadata" aria-label="Sunderland versus Arsenal footage with generated commentary" onTimeUpdate={() => {
          if (playbackWindow.current && player.current && player.current.currentTime >= data.end_s) {
            player.current.pause(); playbackWindow.current = false;
          }
        }} />
        <button className="editor-secondary" onClick={hearLine}>Hear the proposed line in context</button>
        <p className="editor-note">The analyst line starts at {data.start_s.toFixed(1)}s. Playback uses the existing generated voice, with the original broadcast speech removed.</p>
        <details><summary>What informed this line?</summary><blockquote>{data.fact.text}</blockquote><a href={data.fact.url} target="_blank" rel="noreferrer">Read the source passage</a><p>{data.source_origin}</p></details>
      </section>
      <section className="editor-form" aria-labelledby="proposed-title"><h2 id="proposed-title">{session.recorded ? "The review that was saved" : "Would you air this?"}</h2><blockquote className="editor-proposal">“{data.proposed}”</blockquote>
        <div className="editor-decisions" role="group" aria-label="Your decision">
          {(["keep", "rewrite", "reject"] as const).map(action => <button key={action} aria-pressed={decision === action} disabled={busy || !!session.comparison_status} onClick={() => { setDecision(action); change(); }}>{action[0].toUpperCase() + action.slice(1)}</button>)}
        </div>
        {decision === "rewrite" && <div className="editor-field"><label htmlFor="rewrite">Your version</label><textarea id="rewrite" maxLength={500} rows={4} value={rewrite} disabled={busy || !!session.comparison_status} onChange={e => { setRewrite(e.target.value); change(); }} />
          {voiceAvailable && <button className="editor-text-button" onClick={() => preview(rewrite, "edit")}>{speaking === "edit" ? "Playing…" : "Listen to my edit"}</button>}<small>Edited-line previews use this browser’s voice.</small></div>}
        <div className="editor-field"><label htmlFor="reason">Why?</label><textarea id="reason" rows={3} maxLength={1200} value={reason} disabled={busy || !!session.comparison_status} placeholder="What makes this line worth airing, or what needs to change?" onChange={e => { setReason(e.target.value); change(); }} /></div>
        {!session.recorded && <button className="editor-primary" disabled={!loaded || busy || !!session.comparison_status} onClick={save}>{busy ? "Saving and verifying…" : session.connected ? "Save my review to Trajectory" : "Save a preview decision"}</button>}
        {session.review && <p className="editor-saved"><a href={session.recorded ? "/editor-review/evidence.json" : session.review.trajectory_url} target="_blank" rel="noreferrer">{session.recorded ? "Read the recorded evidence" : "Inspect the saved review in Trajectory"}</a></p>}
        {session.review && <button className="editor-secondary" onClick={() => { setStep("compare"); setStatus(""); }}>Test this feedback on another clip</button>}
        {!session.connected && loaded && <p className="editor-note">{session.recorded ? "This is the owner’s actual saved feedback. The next step replays the two model runs it informed." : "This public preview saves decisions in your browser. The owner’s connected local session captures real Trajectory feedback and runs the comparison."}</p>}
      </section>
    </div> : <section className="editor-transfer">
      <div className="editor-section-heading"><h2>{data.transfer.title}</h2><span>A different clip · visual cutoff {data.transfer.cutoff_s}s</span></div>
      <div className="editor-transfer-grid"><video src={data.transfer.video_url} muted controls playsInline preload="metadata" aria-label="Barcelona versus Bayern footage for the new commentary comparison" />
        <div><h3>Does your feedback help here?</h3><p>Both takes use the same model, visual frames, caller context, and factual background. One also receives your saved review from the first clip.</p>
          <p>Your earlier match’s facts are excluded as evidence for this match. The model receives frames only up to {data.transfer.cutoff_s}s.</p>
          {!session.takes && <button className="editor-primary" disabled={busy || !!session.comparison_status} onClick={generate}>{busy ? "Generating two takes…" : "Generate the comparison"}</button>}
          <p className="editor-note">Two capped calls through Trajectory, no retries or training. Your feedback is supplied in context; an improvement is not assumed.</p></div></div>
      <details><summary>Background available to both takes</summary><ul>{data.transfer.research?.map(fact => <li key={fact.id}>{fact.text} <a href={fact.url} target="_blank" rel="noreferrer">Source</a></li>)}</ul><p>{data.transfer.research_limit}</p></details>
      {session.takes && <><div className="editor-takes">{session.takes.map(take => <article key={take.label}><h3>Take {take.label}{session.reveal ? ` · ${session.reveal[take.label] === "with_feedback" ? "With your feedback" : "Baseline"}` : ""}</h3><blockquote>{take.text === "[SILENCE]" ? "Stay silent." : take.text}</blockquote>
        {voiceAvailable && <button className="editor-secondary" onClick={() => preview(take.text, take.label)}>{speaking === take.label ? "Playing…" : `Hear take ${take.label}`}</button>}
        {session.reveal && <a href={session.recorded ? "/editor-review/evidence.json" : take.trajectory_url} target="_blank" rel="noreferrer">Inspect this run</a>}</article>)}</div>
        <p className="editor-note">Listening uses the same browser voice for both takes. These are unchanged model outputs. Take order is randomized.</p>
        <div className="editor-vote"><fieldset><legend>Which would you air?</legend>{["A", "B", "tie", "neither"].map(choice => <label key={choice}><input type="radio" name="take" value={choice} checked={vote === choice} disabled={busy || !!session.vote} onChange={() => setVote(choice)} />{choice === "tie" ? "Equally good" : choice === "neither" ? "Neither" : `Take ${choice}`}</label>)}</fieldset>
          <label htmlFor="vote-reason">What made the difference?</label><textarea id="vote-reason" rows={2} maxLength={1200} value={voteReason} disabled={busy || !!session.vote} onChange={e => setVoteReason(e.target.value)} />
          <button className="editor-primary" disabled={busy || !vote || !voteReason.trim() || !!session.vote} onClick={saveVote}>{busy ? "Saving…" : session.recorded ? "Reveal the runs" : "Save my judgment and reveal the runs"}</button>
          {session.vote && <p>Your judgment: {session.vote.choice}. {session.vote.reason}</p>}
          {session.reveal && session.owner_vote && <p>Original reviewer’s judgment: {session.owner_vote.choice}. {session.owner_vote.reason}</p>}</div></>}
    </section>}
    <div className="editor-feedback" aria-live="polite">{status && <p>{status}</p>}{error && <p role="alert" className="editor-error">{error}</p>}</div>
    <footer className="editor-footer"><p>Learning through retained human feedback in the agent’s context. No model weights are updated. A single comparison does not establish general improvement.</p><a href="https://www.aarshroongta.com/projects/soccer-commentary">About the commentary project</a></footer>
  </main>;
}
