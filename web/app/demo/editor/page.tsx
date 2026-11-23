import type { Metadata } from "next";
import data from "@/data/editor-review.json";
import Editor from "./review";
import "./review.css";

export const metadata: Metadata = {
  title: "Commentary review | Aarsh Roongta",
  description: "Review a commentary line and compare two takes.",
};

export default function EditorPage() { return <Editor data={data} />; }
