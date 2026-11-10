"use client";

import { useEffect, useId, useRef, useState } from "react";

type Diagram = { title: string; description: string; source: string };

export default function MermaidDiagram({ diagram, theme, compact = false }: {
  diagram: Diagram; theme: "light" | "dark"; compact?: boolean;
}) {
  const id = `mermaid-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const target = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function draw() {
      try {
        const { default: mermaid } = await import("mermaid");
        await document.fonts.ready;
        if (cancelled) return;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          look: "classic",
          fontFamily: "var(--font-project), system-ui, sans-serif",
          themeVariables: {
            darkMode: theme === "dark",
            textColor: theme === "dark" ? "#ededf0" : "#16161a",
            primaryColor: theme === "dark" ? "#1a2032" : "#f3f5fc",
            primaryTextColor: theme === "dark" ? "#ededf0" : "#16161a",
            primaryBorderColor: theme === "dark" ? "#687aac" : "#8d9ccc",
            lineColor: theme === "dark" ? "#a2a2ae" : "#62626c",
            secondaryColor: theme === "dark" ? "#151618" : "#fafafb",
            tertiaryColor: theme === "dark" ? "#151618" : "#fafafb",
            edgeLabelBackground: theme === "dark" ? "#0f1011" : "#ffffff",
            fontSize: "13px",
          },
          flowchart: { htmlLabels: false, curve: "linear", nodeSpacing: 24, rankSpacing: 28, padding: 12 },
        });
        const { svg } = await mermaid.render(`${id}-${theme}`, diagram.source);
        if (!cancelled && target.current) {
          // The diagram source is a checked-in constant, never user or model input.
          target.current.innerHTML = svg;
          target.current.querySelector("svg")?.setAttribute("aria-label", diagram.description);
          setFailed(false);
        }
      } catch {
        if (!cancelled) {
          if (target.current) target.current.innerHTML = "";
          setFailed(true);
        }
      }
    }
    void draw();
    return () => { cancelled = true; };
  }, [diagram.source, diagram.description, id, theme]);

  return <figure className={`engineering-diagram${compact ? " engineering-diagram-compact" : ""}`}>
    <figcaption>{diagram.title}<span>Mermaid</span></figcaption>
    <div className="engineering-diagram-scroll" tabIndex={0} role="region" aria-label={diagram.description}>
      <div ref={target} className="engineering-diagram-canvas">
        <p className="engineering-diagram-loading">Loading flowchart…</p>
      </div>
    </div>
    {failed && <p className="engineering-diagram-error">The visual could not load. Read the flowchart source below.</p>}
    <details className="engineering-diagram-source"><summary>View Mermaid source</summary><pre><code>{diagram.source}</code></pre></details>
  </figure>;
}
