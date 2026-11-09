import type { Metadata } from "next";
import { Geist, Geist_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const projectFont = Space_Grotesk({
  variable: "--font-project",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Football Commentary Agents — Aarsh Roongta",
  description:
    "Three football commentary examples, independent agent roles, inspectable traces and a small evaluation case study. A project by Aarsh Roongta.",
  openGraph: {
    title: "Football Commentary Agents — Aarsh Roongta",
    description: "A portfolio project in multimodal perception, researched context and coordinated AI agents.",
    type: "website",
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${projectFont.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-ink text-text">{children}</body>
    </html>
  );
}
