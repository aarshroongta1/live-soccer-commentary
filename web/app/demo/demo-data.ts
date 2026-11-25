import { existsSync } from "node:fs";
import { join } from "node:path";
import agentShowcase from "@/data/agent-showcase.json";
import savedRuns from "@/data/agent-runs.json";
import featured from "@/data/featured-demos.json";
import prepared from "@/data/portfolio-demo.json";
import iterationData from "@/data/iteration-demo.json";
import { type DemoClip } from "./engineering-experience";

const localMedia = (path: string) => existsSync(join(process.cwd(), "public", path.slice(1))) ? path : null;

export function getDemoData(showcase = false) {
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
  const showcaseClips: DemoClip[] = agentShowcase.map((clip) => ({
    ...clip,
    kind: "automatic",
    subtitle: "Agent-generated caller and analyst commentary",
    note: "Selected agent output on saved perception. Generated words unchanged; recorded voices and source excerpts.",
    src: localMedia(clip.videoPath),
    poster: localMedia(clip.posterPath),
  }));
  return { clips: showcase ? showcaseClips : clips, runs, preparations, iteration };
}
