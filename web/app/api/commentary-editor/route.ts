import { spawn } from "node:child_process";
import { homedir } from "node:os";
import { resolve } from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
let active = false;

function local(request: Request) {
  if (process.env.VERCEL || process.env.NODE_ENV !== "development") return false;
  const host = request.headers.get("host") || "";
  return /^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(host);
}

async function worker(body: object): Promise<Record<string, unknown>> {
  const root = resolve(process.cwd(), "..");
  return new Promise((done) => {
    const child = spawn(resolve(homedir(), ".local/share/uv/tools/trajectory-sdk/bin/python"),
      ["scripts/commentary_editor.py"], { cwd: root, stdio: ["pipe", "pipe", "pipe"] });
    let output = "";
    const timer = setTimeout(() => child.kill("SIGTERM"), 110_000);
    child.stdout.on("data", chunk => { output += chunk.toString(); });
    child.stderr.on("data", () => { /* Credentials and provider errors never enter the browser. */ });
    child.on("error", () => { clearTimeout(timer); done({ error: "Could not start the connected worker." }); });
    child.on("close", () => {
      clearTimeout(timer);
      try { done(JSON.parse(output)); } catch { done({ error: "Worker stopped. Retained runs will not be regenerated automatically." }); }
    });
    child.stdin.end(JSON.stringify(body));
  });
}

export async function GET(request: Request) {
  if (!local(request)) return Response.json({ connected: false }, { headers: { "Cache-Control": "no-store" } });
  return Response.json(await worker({ action: "status" }), { headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: Request) {
  const origin = `${new URL(request.url).protocol}//${request.headers.get("host")}`;
  if (!local(request) || request.headers.get("origin") !== origin) {
    return Response.json({ error: "Live capture is available in the connected local player." }, { status: 403 });
  }
  if (active) return Response.json({ error: "A save or comparison is already in progress." }, { status: 409 });
  const text = await request.text();
  if (text.length > 5000) return Response.json({ error: "Review is too long." }, { status: 413 });
  let body;
  try { body = JSON.parse(text); } catch { return Response.json({ error: "Invalid request." }, { status: 400 }); }
  if (!["review", "compare", "vote"].includes(body?.action)) return Response.json({ error: "Invalid action." }, { status: 400 });
  active = true;
  try {
    const result = await worker(body);
    return Response.json(result, { status: result.error ? 400 : 200 });
  } finally { active = false; }
}
