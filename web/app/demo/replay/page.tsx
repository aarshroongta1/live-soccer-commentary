import trace from "@/public/soccer-commentary/prepared-penalty.trace.json";
import delivery from "@/data/replay-delivery.json";
import Replay from "./replay";
import type { Metadata } from "next";
import comparison from "@/data/search-comparison.json";
import Comparison from "../keenable/comparison";

export const metadata: Metadata = { title: "Run inspector | Commentary", description: "Inspect saved commentary turns, visual evidence, identity tools and raw agent payloads alongside match audio." };

export default async function ReplayPage({ searchParams }: PageProps<"/demo/replay">) {
  const query = await searchParams;
  if (query.run === "penalty") return <Replay trace={trace} delivery={delivery.segments} />;
  return <Comparison data={comparison} mode={query.mode === "replay" ? "replay" : "live"} />;
}
