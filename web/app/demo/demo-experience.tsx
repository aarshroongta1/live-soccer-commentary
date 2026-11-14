"use client";

import { useEffect, useRef, useState } from "react";
import type { DemoClip } from "./engineering-experience";
import "./demo.css";
import "./showcase.css";

export type { DemoClip } from "./engineering-experience";

function ShowcaseVideo({ clip }: { clip: DemoClip }) {
  const video = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const player = video.current;
    return () => { player?.pause(); };
  }, []);

  return (
    <>
      {clip.src && !failed ? <video
        ref={video} src={clip.src} poster={clip.poster ?? undefined}
        controls playsInline preload="metadata"
        aria-label={`${clip.title}: AI commentary demo with two voices and captions`}
        onError={() => setFailed(true)}
      >Your browser cannot play this video.</video> : <div className="showcase-missing" role="status">
        <h2>Video unavailable</h2><p>Here’s the commentary for this clip.</p>
        <ul>{clip.lines.map((line, index) => <li key={index}><strong>{line.voice === "caller" ? "Caller" : "Analyst"}:</strong> {line.text}</li>)}</ul>
      </div>}
      <div className="showcase-caption">
        <span>{clip.duration_s}s <span aria-hidden="true">/</span> Captioned demo <span aria-hidden="true">/</span> AI voices</span>
      </div>
    </>
  );
}

export default function DemoExperience({ clips }: { clips: DemoClip[] }) {
  const [activeId, setActiveId] = useState(clips[0].id);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const active = clips.find((clip) => clip.id === activeId)!;
  return (
    <div className="demo-page showcase-page" data-theme={theme}>
      <main className="showcase-content">
        <nav className="showcase-nav" aria-label="Project navigation">
          <a className="showcase-author" href="https://www.aarshroongta.com">Aarsh Roongta<span aria-hidden="true">.</span></a>
          <div><a href="https://www.aarshroongta.com/projects">Projects</a><a href="https://github.com/aarshroongta1/live-soccer-commentary" target="_blank" rel="noreferrer">GitHub ↗</a>
            <button className="showcase-theme" type="button" aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`} aria-pressed={theme === "dark"} onClick={() => setTheme(theme === "light" ? "dark" : "light")}>{theme === "light" ? "◐" : "◑"}</button>
          </div>
        </nav>
        <header className="showcase-heading">
          <h1>AI football commentary<span aria-hidden="true">.</span></h1>
          <p>Two AI commentators follow the action and explain the play.</p>
        </header>
        <section aria-label="Football commentary demo">
          <div className="showcase-clips" role="group" aria-label="Choose a demo clip">
            {clips.map((clip) => <button key={clip.id} type="button" aria-pressed={activeId === clip.id} onClick={() => setActiveId(clip.id)}><span>{clip.title}</span><span>{clip.duration_s}s</span></button>)}
          </div>
          <ShowcaseVideo key={active.id} clip={active} />
        </section>
        <footer className="showcase-footer"><span>Recorded examples</span><a href="https://github.com/aarshroongta1/live-soccer-commentary#architecture" target="_blank" rel="noreferrer">How I built it <span aria-hidden="true">↗</span></a>{active.src && <a href={active.src} download>Download clip</a>}</footer>
      </main>
    </div>
  );
}
