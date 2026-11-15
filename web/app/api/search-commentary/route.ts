import { spawn } from "node:child_process";
import { resolve } from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
let active = false;
let runs = 0;

export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");
  if (!host || !/^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(host) ||
      origin !== `${new URL(request.url).protocol}//${host}`) {
    return Response.json({ error: "localhost only" }, { status: 403 });
  }
  if (active) return Response.json({ error: "A search run is already active." }, { status: 409 });
  if (runs >= 3) return Response.json({ error: "Development limit reached: three live runs per server session." }, { status: 429 });
  const body = await request.json().catch(() => null);
  if (body?.spend !== true) return Response.json({ error: "Explicit spend flag required." }, { status: 400 });
  active = true; runs += 1;
  const root = resolve(process.cwd(), "..");
  const child = spawn(resolve(root, ".venv/bin/python"), ["scripts/live_search_commentary.py", "--spend"], {
    cwd: root, stdio: ["ignore", "pipe", "pipe"],
  });
  const encoder = new TextEncoder();
  let closed = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      child.stdout.on("data", chunk => { if (!closed) controller.enqueue(new Uint8Array(chunk)); });
      child.stderr.on("data", () => { /* Provider errors stay out of the browser. */ });
      child.on("error", () => {
        if (!closed) controller.enqueue(encoder.encode(JSON.stringify({ type: "pipeline_error", message: "Could not start the local research worker." }) + "\n"));
      });
      child.on("close", code => {
        active = false;
        if (!closed) {
          if (code) controller.enqueue(encoder.encode(JSON.stringify({ type: "pipeline_error", message: "Research worker stopped. Check the local run audit." }) + "\n"));
          closed = true; controller.close();
        }
      });
      request.signal.addEventListener("abort", () => { closed = true; child.kill("SIGTERM"); }, { once: true });
    },
    cancel() { closed = true; child.kill("SIGTERM"); },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store", "X-Accel-Buffering": "no" } });
}
