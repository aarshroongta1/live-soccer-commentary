const stages = [
  {
    title: "Clip + match context",
    stack: "OpenCV",
    detail: "Sampled frames, with an optional roster and pre-match research.",
  },
  {
    title: "Observe + write",
    stack: "Python · LangGraph · OpenAI / Claude",
    detail: "Visual observations feed one shared caller–analyst script.",
  },
  {
    title: "Check + edit",
    stack: "Pydantic · manual review",
    detail: "Check evidence references and timing, then review the words.",
  },
  {
    title: "Caller + analyst",
    stack: "ElevenLabs",
    detail: "Two custom voices, using their saved voice settings.",
  },
  {
    title: "Video + captions",
    stack: "FFmpeg",
    detail: "Align the measured speech and captions with the footage.",
  },
  {
    title: "Saved web demo",
    stack: "Next.js / React · TypeScript · Vercel",
    detail: "Play the video, switch clips, and explore the transcript.",
  },
];

export default function CommentaryArchitecture() {
  return (
    <section className="demo-architecture" aria-labelledby="demo-architecture-title">
      <h2 id="demo-architecture-title">From footage to commentary</h2>
      <p className="demo-architecture-intro">
        The recorded workflow, with the stack behind each stage.
      </p>
      <ol className="demo-flow" aria-label="Recorded commentary pipeline">
        {stages.map((stage, index) => (
          <li className="demo-flow-stage" key={stage.title}>
            <span className="demo-flow-number" aria-hidden="true">{index + 1}</span>
            <h3>{stage.title}</h3>
            <p className="demo-flow-stack">{stage.stack}</p>
            <p className="demo-flow-detail">{stage.detail}</p>
            {index < stages.length - 1 && (
              <span className="demo-flow-arrow" aria-hidden="true">→</span>
            )}
          </li>
        ))}
      </ol>
      <p className="demo-architecture-note">
        These showcase scripts are manually reviewed and edited before voicing.
        Playback uses saved videos and makes no model calls.
      </p>
    </section>
  );
}
