const stages = [
  { title: "Pre-match researcher", stack: "Python · cited research", detail: "Prepares cited squads, shirt numbers, team context and optional stats before replay. Research cannot establish on-pitch events." },
  { title: "Visual observer", stack: "OpenCV · vision model", detail: "Reads available frames. Can inspect a region and request research_roster, then propose an action-linked identity." },
  { title: "Shared evidence", stack: "Pydantic · read-only tools", detail: "Stores timestamped evidence and agent-proposed identities. Code checks citations, availability and roster consistency." },
  { title: "Caller + analyst", stack: "Independent LangGraph turns", detail: "Separate prompts, model choices and tool loops. Each can speak, request context, or abstain." },
  { title: "Deterministic coordinator", stack: "Python asyncio", detail: "Owns the replay clock, deadlines, turn-taking and goal priority. Rejects stale or unavailable evidence." },
  { title: "Text + run trace", stack: "Next.js · React · TypeScript", detail: "Saved observations, exact generated words and execution events. Replay speech timing is simulated." },
];

export default function CommentaryArchitecture() {
  return (
    <section className="demo-architecture" aria-labelledby="demo-architecture-title">
      <h2 id="demo-architecture-title">How the agents work</h2>
      <p className="demo-architecture-intro">Independent roles share evidence. A deterministic coordinator controls when they speak.</p>
      <ol className="demo-flow" aria-label="Multi-agent commentary architecture">
        {stages.map((stage, index) => (
          <li className="demo-flow-stage" key={stage.title}>
            <span className="demo-flow-number" aria-hidden="true">{index + 1}</span>
            <h3>{stage.title}</h3>
            <p className="demo-flow-stack">{stage.stack}</p>
            <p className="demo-flow-detail">{stage.detail}</p>
            {index < stages.length - 1 && <span className="demo-flow-arrow" aria-hidden="true">→</span>}
          </li>
        ))}
      </ol>
      <p className="demo-architecture-note">The prototype also supports a bounded reviewer/revision step, disabled by default. Saved examples below show which roles and tools actually ran; research is optional.</p>
      <details className="demo-recorded-path">
        <summary>Recorded preparation and causal replay</summary>
        <p>The England example reuses saved perception and research. Arsenal and Argentina use fresh full-context vision and separate caller/analyst turns, followed by disclosed source cuts and caption edits. All three are prepared before playback. The separate causal replay prototype releases frames over time and simulates speech occupancy. None of these examples establishes real-time spoken commentary latency.</p>
      </details>
    </section>
  );
}
