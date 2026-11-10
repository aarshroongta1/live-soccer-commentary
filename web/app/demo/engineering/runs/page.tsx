import { getDemoData } from "../../demo-data";
import EngineeringExperience from "../../engineering-experience";

export const metadata = { title: "Saved runs — Football Commentary Agents" };

export default function RunsPage() {
  return <EngineeringExperience {...getDemoData()} />;
}
