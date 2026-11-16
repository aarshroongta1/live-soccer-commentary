import type { Metadata } from "next";
import comparison from "@/data/search-comparison.json";
import Comparison from "./comparison";

export const metadata: Metadata = {
  title: "Soccer commentary | Keenable & Exa",
  description: "A football commentary workflow comparing Keenable and Exa search results and response times.",
};

export default function KeenableDemo() {
  return <Comparison data={comparison} mode="replay" />;
}
