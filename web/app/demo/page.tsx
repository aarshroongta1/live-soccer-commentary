import { existsSync } from "node:fs";
import { join } from "node:path";
import savedRuns from "@/data/agent-runs.json";
import catalog from "@/data/demo-clips.json";
import prepared from "@/data/portfolio-demo.json";
import DemoExperience, { type DemoClip } from "./demo-experience";

const localMedia = (path: string) => existsSync(join(process.cwd(), "public", path.slice(1))) ? path : null;

export default function DemoPage() {
  const clips: DemoClip[] = [{
    ...prepared,
    kind: "automatic",
    src: process.env.PORTFOLIO_PENALTY_VIDEO_URL || localMedia(prepared.videoPath),
    poster: localMedia(prepared.posterPath),
  }];
  for (const id of ["arsenal-norwich", "argentina-france"]) {
    const clip = catalog.find((item) => item.id === id)!;
    clips.push({
      ...clip,
      kind: "curated",
      title: id === "arsenal-norwich" ? "Arsenal vs Norwich" : "Argentina vs France",
      subtitle: id === "arsenal-norwich" ? "A one-touch team goal" : "A counterattack in the final",
      src: process.env[`PORTFOLIO_${id === "arsenal-norwich" ? "ARSENAL" : "ARGENTINA"}_VIDEO_URL`] || localMedia(`/soccer-commentary/portfolio-${id}.mp4`),
      poster: localMedia(`/soccer-commentary/portfolio-${id}.jpg`),
      note: "Manually reviewed and edited showcase. Captions preserve the retained script; this is not unedited multi-agent output.",
    });
  }
  const runs = savedRuns.map((run) => ({ ...run, src: localMedia(run.videoPath) }));
  return <DemoExperience clips={clips} runs={runs} preparation={prepared} />;
}
