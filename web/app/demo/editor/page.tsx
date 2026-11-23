import type { Metadata } from "next";
import data from "@/data/editor-review.json";
import Editor from "./review";
import "./review.css";

export const metadata: Metadata = {
  title: "Teach the commentary agent | Aarsh Roongta",
  description: "Watch the play, keep or correct a generated line, and test your feedback on another clip.",
};

export default function EditorPage() { return <Editor data={data} />; }
