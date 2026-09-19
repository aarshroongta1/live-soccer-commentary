import { NextResponse } from "next/server";
import * as worker from "@/lib/live-worker";

export const runtime = "nodejs";
const error = (message: string, status = 400) => NextResponse.json({ error: message }, { status });
function local(request: Request) {
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");
  return Boolean(origin && host && origin === `${new URL(request.url).protocol}//${host}` && /^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(host));
}

export async function POST(request: Request) {
  if (!local(request)) return error("localhost only", 403);
  if (Number(request.headers.get("content-length") ?? 0) > 6_000_000) return error("request too large", 413);
  try {
    const reader = request.body?.getReader();
    let bytes = 0; const chunks: Uint8Array[] = [];
    if (!reader) return error("request body required");
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 6_000_000) { await reader.cancel(); return error("request too large", 413); }
      chunks.push(value);
    }
    const body = JSON.parse(new TextDecoder().decode(Buffer.concat(chunks)));
    if (!body || typeof body !== "object") return error("invalid live request");
    if (body.action === "start" && body.spend === true && (body.context === "argfra" || body.context === "generic")) return NextResponse.json(await worker.start(body.context));
    if (body.action === "batch" && typeof body.session_id === "string" && typeof body.playback_s === "number" && Array.isArray(body.frames)) return NextResponse.json(await worker.batch(body.session_id, body.playback_s, body.frames));
    if (body.action === "stop" && typeof body.session_id === "string") return NextResponse.json(worker.end(body.session_id));
    return error("invalid live request");
  } catch (cause) {
    // Fixed lifecycle messages only: never expose provider text or image payloads.
    const messages: Record<string, [string, number]> = {
      "worker is busy": ["A batch is still processing.", 409],
      "a live session is already active": ["Stop the other session, or wait 90 seconds.", 409],
      "unknown session": ["Session ended. Start a new session.", 410],
      "session limit reached": ["Session limit reached.", 410],
      "worker rejected start": ["Check OPENAI_API_KEY and the selected match context files.", 503],
      "worker rejected batch": ["Model or frame validation failed. Session stopped; try again.", 422],
      "worker timed out": ["Model timed out. Session stopped; try again.", 504],
    };
    const message = cause instanceof Error ? messages[cause.message] : undefined;
    return message ? error(...message) : error("Live request stopped. Check the server and retry.", 503);
  }
}
