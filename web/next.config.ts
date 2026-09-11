import type { NextConfig } from "next";

/**
 * The Python runtime owns every byte this page shows: the delayed MJPEG, the
 * event stream, the state snapshot. Proxying them through Next means the
 * browser only ever talks to one origin, so there is no CORS layer to keep in
 * sync with the FastAPI app.
 *
 * The rewrites below cover the request/response endpoints. The two streaming
 * ones are route handlers instead (`app/api/stream`, `app/api/video`) because
 * the proxy gzips a rewritten response and a gzipped event stream never
 * reaches the browser — see `lib/proxy.ts`. Route handlers win over rewrites
 * on the same path, so these entries are the fallback for everything else.
 */
const API_ORIGIN = process.env.COMMENTARY_API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // The dev badge sits in the bottom-left corner, over the transcript. This
  // page is demoed from `npm run dev` more often than from a build.
  devIndicators: false,

  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` },
      { source: "/healthz", destination: `${API_ORIGIN}/healthz` },
    ];
  },
};

export default nextConfig;
