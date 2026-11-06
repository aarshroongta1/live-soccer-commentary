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
  title: "Live Soccer Commentary AI — Demo",
  description:
    "A recorded demo of two-voice soccer commentary generated from video frames and pre-match research. A project by Aarsh Roongta.",
  openGraph: {
    title: "Live Soccer Commentary AI — Demo",
    description: "A 40-second recorded demo of AI-generated soccer commentary.",
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
