import { getDemoData } from "../demo-data";
import EngineeringExperience from "../engineering-experience";

export const metadata = { title: "How I built it — Football Commentary Agents" };

export default function EngineeringPage() {
  return <EngineeringExperience {...getDemoData()} />;
}
