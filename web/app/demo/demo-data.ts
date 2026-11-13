import { existsSync } from "node:fs";
import { join } from "node:path";
import editedShowcase from "@/data/edited-showcase.json";
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
    src: process.env.PORTFOLIO_PENALTY_VIDEO_URL || localMedia(showcase ? "/soccer-commentary/showcase-penalty-voiced.mp4" : prepared.videoPath),
    poster: localMedia(prepared.posterPath),
  }];
  for (const clip of featured) {
    clips.push({
      ...clip,
      kind: "lightly-edited",
      src: process.env[`PORTFOLIO_${clip.id === "generated-arsenal" ? "ARSENAL" : "ARGENTINA"}_VIDEO_URL`] || localMedia(showcase ? `/soccer-commentary/showcase-${clip.id === "generated-arsenal" ? "arsenal" : "argentina"}-voiced.mp4` : clip.videoPath),
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
  const showcaseClips: DemoClip[] = editedShowcase.map((clip) => ({
    ...clip,
    kind: "curated",
    subtitle: "Edited caller and analyst commentary",
    note: "Editorial rewrite for the showcase; original agent outputs retained separately.",
    src: localMedia(clip.videoPath),
    poster: localMedia(clip.posterPath),
  }));
  return { clips: showcase ? showcaseClips : clips, runs, preparations, iteration };
}
