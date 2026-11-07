import { existsSync } from "node:fs";
import { join } from "node:path";
import savedRuns from "@/data/agent-runs.json";
import catalog from "@/data/demo-clips.json";
import DemoExperience from "./demo-experience";

export default function DemoPage() {
  const clips = catalog.map((clip) => ({
    id: clip.id,
    title: clip.title,
    lines: clip.lines,
    src:
      process.env[clip.envKey] ||
      (existsSync(join(process.cwd(), "public", clip.videoPath.slice(1)))
        ? clip.videoPath
        : null),
  }));
  const runs = savedRuns.map((run) => ({
    ...run,
    src: existsSync(join(process.cwd(), "public", run.videoPath.slice(1)))
      ? run.videoPath : null,
  }));
  return <DemoExperience clips={clips} runs={runs} />;
}
