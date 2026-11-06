"use client";

import { useEffect, useRef, useState } from "react";
import "./demo.css";

export type DemoClip = {
  id: string;
  title: string;
  src: string | null;
  lines: { at_s: number; voice: string; text: string }[];
};
const timestamp = (seconds: number) =>
  `0:${Math.floor(seconds).toString().padStart(2, "0")}`;

function ClipPlayer({ clip }: { clip: DemoClip }) {
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  const [selected, setSelected] = useState(0);
  const [failed, setFailed] = useState(false);
  const available = Boolean(clip.src) && !failed;

  useEffect(() => {
    const player = video.current;
    return () => {
      player?.pause();
    };
  }, []);

  return (
    <>
      {available ? (
        <video
          ref={video}
          src={clip.src!}
          controls
          playsInline
          preload="metadata"
          aria-label={`${clip.title} with AI commentary`}
          onTimeUpdate={(event) => {
            const time = event.currentTarget.currentTime;
            setSelected(
              Math.max(
                0,
                clip.lines.findLastIndex((line) => line.at_s <= time),
              ),
            );
          }}
          onLoadedMetadata={(event) => {
            if (pendingSeek.current !== null) {
              event.currentTarget.currentTime = pendingSeek.current;
              pendingSeek.current = null;
            }
          }}
          onError={() => setFailed(true)}
        >
          Your browser cannot play this video. The transcript is available
          below.
        </video>
      ) : (
        <p className="demo-missing" role="status">
          This video couldn’t be loaded. Choose another clip or read its
          transcript below.
        </p>
      )}

      <p className="demo-tech">
        Python and LangGraph coordinate video observation, pre-match research,
        and two-voice script writing. These recorded showcase scripts are
        reviewed and edited before voicing; they are not unedited system output.
        ElevenLabs or local voices handle speech, with FFmpeg aligning audio
        and captions.
      </p>

      <details className="demo-transcript" open={!available}>
        <summary>Transcript</summary>
        <ol className="demo-lines" aria-label={`${clip.title} transcript`}>
          {clip.lines.map((line, index) => (
            <li key={index}>
              <button
                type="button"
                aria-current={selected === index ? "true" : undefined}
                onClick={() => {
                  setSelected(index);
                  if (video.current && available) {
                    if (video.current.readyState === 0) pendingSeek.current = line.at_s;
                    else video.current.currentTime = line.at_s;
                  }
                }}
              >
                <time>{timestamp(line.at_s)}</time>
                <span className="demo-voice">
                  {line.voice === "caller" ? "Caller" : "Analyst"}
                </span>
                <span>{line.text}</span>
              </button>
            </li>
          ))}
        </ol>
      </details>
    </>
  );
}

export default function DemoExperience({ clips }: { clips: DemoClip[] }) {
  const [activeId, setActiveId] = useState(clips[0].id);
  const active = clips.find((clip) => clip.id === activeId)!;

  return (
    <main className="demo-page">
      <div className="demo-content">
        <h1 className="demo-sr-only">Soccer commentary demos</h1>
        <nav className="demo-nav" aria-label="Project navigation">
          <a href="https://www.aarshroongta.com/projects">Back to projects</a>
          <a
            href="https://github.com/aarshroongta1/live-soccer-commentary"
            target="_blank"
            rel="noreferrer"
          >
            Source on GitHub
          </a>
        </nav>
        <label className="demo-picker">
          <span className="demo-sr-only">Select a video</span>
          <select
            value={activeId}
            onChange={(event) => setActiveId(event.target.value)}
          >
            {clips.map((clip) => (
              <option key={clip.id} value={clip.id}>
                {clip.title}
              </option>
            ))}
          </select>
        </label>
        <ClipPlayer key={active.id} clip={active} />
      </div>
    </main>
  );
}
