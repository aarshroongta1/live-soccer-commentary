import { existsSync } from "node:fs";
import { join } from "node:path";
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
  return <DemoExperience clips={clips} />;
}
