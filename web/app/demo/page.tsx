import { getDemoData } from "./demo-data";
import DemoExperience from "./demo-experience";

export default function DemoPage() {
  const { clips } = getDemoData(true);
  const featured = clips.map(({ id, title, subtitle, src, poster, duration_s, kind, note, lines }) =>
    ({ id, title, subtitle, src, poster, duration_s, kind, note, lines }));
  return <DemoExperience clips={featured} />;
}
