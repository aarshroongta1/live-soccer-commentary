import { existsSync } from "node:fs";
import { join } from "node:path";
import savedRuns from "@/data/agent-runs.json";
import featured from "@/data/featured-demos.json";
import prepared from "@/data/portfolio-demo.json";
import iterationData from "@/data/iteration-demo.json";
import DemoExperience, { type DemoClip } from "./demo-experience";

const localMedia = (path: string) => existsSync(join(process.cwd(), "public", path.slice(1))) ? path : null;

export default function DemoPage() {
  const clips: DemoClip[] = [{
    ...prepared,
    kind: "automatic",
    src: process.env.PORTFOLIO_PENALTY_VIDEO_URL || localMedia(prepared.videoPath),
    poster: localMedia(prepared.posterPath),
  }];
  for (const clip of featured) {
    clips.push({
      ...clip,
      kind: "lightly-edited",
      src: process.env[`PORTFOLIO_${clip.id === "generated-arsenal" ? "ARSENAL" : "ARGENTINA"}_VIDEO_URL`] || localMedia(clip.videoPath),
      poster: localMedia(clip.posterPath),
    });
  }
  const preparations = Object.fromEntries([prepared, ...featured].map((clip) => [clip.id, clip]));
  const runs = savedRuns.map((run) => ({ ...run, src: localMedia(run.videoPath) }));
  const iteration = {
    ...iterationData,
    before: {
      ...iterationData.before,
      clip: {
        ...iterationData.before.clip,
        kind: "automatic" as const,
        src: process.env.PORTFOLIO_BEFORE_VIDEO_URL || localMedia(iterationData.before.clip.videoPath),
        poster: localMedia(iterationData.before.clip.posterPath),
      },
    },
  };
  return <DemoExperience clips={clips} runs={runs} preparations={preparations} iteration={iteration} />;
}
