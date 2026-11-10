const stages = [
  { title: "Observe the frames", file: "recorded.py", detail: "The visual observer proposes timestamped actions and supporting frames. Unclear actions can remain uncertain." },
  { title: "Share the evidence", file: "contracts.py", detail: "Typed observations and optional squad context become shared input. A citation points to an observation; it does not verify the pixels." },
  { title: "Coordinate two roles", file: "agents.py", detail: "The caller reacts; the analyst explains. Code controls evidence visibility, word budgets and turn-taking. Each agent may abstain." },
  { title: "Save text and trace", file: "render_text_demo.py", detail: "Accepted lines become source-timed captions. Model outputs, rejected turns and costs remain available in the saved run." },
];

export default function CommentaryArchitecture() {
  return (
    <section className="demo-architecture" id="pipeline" aria-labelledby="demo-architecture-title">
      <div className="demo-section-heading"><h2 id="demo-architecture-title">A small, inspectable pipeline</h2><span>Models choose content. Code enforces rules.</span></div>
      <ol className="demo-flow" aria-label="Multi-agent commentary architecture">
        {stages.map((stage, index) => <li className="demo-flow-stage" key={stage.title}>
          <span className="demo-flow-number" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
          <div><h3>{stage.title}</h3><p>{stage.detail}</p><code>{stage.file}</code></div>
        </li>)}
      </ol>
      <details className="demo-recorded-path"><summary>What runs in these examples?</summary>
        <p>The England example reuses saved perception and cited squad context. Arsenal and Argentina use fresh full-context vision and separate caller/analyst turns, followed by disclosed source cuts and caption edits. All three are prepared before playback. The optional reviewer is disabled in the selected runs.</p>
        <p>The separate causal replay prototype releases frames over time, supports bounded context tools, and simulates speech occupancy. These recorded examples do not establish real-time spoken commentary latency.</p>
      </details>
    </section>
  );
}
