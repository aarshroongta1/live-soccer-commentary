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
  title: "AI Football Commentary — Aarsh Roongta",
  description:
    "Watch AI-generated play-by-play and analysis on three short football clips. A project by Aarsh Roongta.",
  openGraph: {
    title: "AI Football Commentary — Aarsh Roongta",
    description: "AI-generated football commentary: a caller follows the action and an analyst explains the play.",
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
