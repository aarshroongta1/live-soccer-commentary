export default function DemoPage() {
  return (
    <main className="min-h-screen px-4 py-6 sm:px-6 sm:py-8">
      <div className="max-w-4xl">
        <header className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3 border-b border-line pb-3">
          <div>
            <h1 className="font-mono text-[11px] uppercase tracking-[0.28em] text-text">
              Argentina–France
            </h1>
            <p className="mt-1 text-sm text-dim">One-off ElevenLabs preview</p>
          </div>
          <a href="/live" className="text-sm underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller">
            Try screen commentary
          </a>
        </header>

        <video
          controls
          playsInline
          preload="metadata"
          className="block aspect-video w-full bg-panel"
          aria-label="Argentina–France recorded commentary preview"
        >
          <source src="/argfra-elevenlabs-preview.mp4" type="video/mp4" />
          Your browser cannot play this preview. {" "}
          <a href="/argfra-elevenlabs-preview.mp4">Download the MP4 preview.</a>
        </video>

        <div className="mt-4 space-y-1.5 text-sm leading-6 text-dim">
          <p>
            <span className="text-caller">ElevenLabs caller</span> · {" "}
            <span className="text-analyst">ElevenLabs analyst</span> · captions embedded
          </p>
          <p>Saved audio preview. The newer names-priority script has not been voiced yet.</p>
          <p>
            <a
              href="/argfra-current-preview.mp4"
              className="underline decoration-line underline-offset-4 hover:text-text focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-caller"
            >
              Original local-voice preview
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}
