import { randomUUID } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { join } from "node:path";
import { createInterface } from "node:readline";

const ROOT = join(process.cwd(), "..");
const LIMITS = { duration_s: 60, max_batches: 12, max_cost_usd: 0.35 };

type Reply = { ok: boolean; result?: Record<string, unknown>; error?: string };
type Pending = {
  reject: (reason: Error) => void;
  resolve: (reply: Reply) => void;
  timer: ReturnType<typeof setTimeout>;
};
type State = {
  batches: number;
  child: ChildProcessWithoutNullStreams;
  cost: number;
  expiry: ReturnType<typeof setTimeout>;
  id: string;
  pending?: Pending;
};

declare global {
  var __liveWorker: State | undefined;
}

function close(state: State, reason = new Error("worker stopped")): void {
  if (globalThis.__liveWorker === state) globalThis.__liveWorker = undefined;
  clearTimeout(state.expiry);
  if (state.pending) {
    clearTimeout(state.pending.timer);
    state.pending.reject(reason);
    state.pending = undefined;
  }
  if (!state.child.killed) state.child.kill();
}

function command(state: State, message: Record<string, unknown>): Promise<Reply> {
  if (state.pending) return Promise.reject(new Error("worker is busy"));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => close(state, new Error("worker timed out")), 45000);
    state.pending = { resolve, reject, timer };
    state.child.stdin.write(`${JSON.stringify(message)}\n`, (writeError) => {
      if (writeError) close(state, new Error("worker unavailable"));
    });
  });
}

function pythonCommand(): { args: string[]; command: string } {
  const python = process.env.VIRTUAL_ENV ? join(process.env.VIRTUAL_ENV, "bin/python") : "uv";
  return python === "uv"
    ? { command: python, args: ["run", "python", "scripts/stream_worker.py"] }
    : { command: python, args: ["scripts/stream_worker.py"] };
}

export async function start(context: "argfra" | "generic") {
  if (globalThis.__liveWorker) throw new Error("a live session is already active");
  const { command: executable, args } = pythonCommand();
  const child = spawn(executable, args, { cwd: ROOT, stdio: "pipe" });
  const state: State = {
    id: randomUUID(),
    child,
    batches: 0,
    cost: 0,
    expiry: setTimeout(() => close(state, new Error("session expired")), 90000),
  };
  globalThis.__liveWorker = state;
  createInterface({ input: child.stdout }).on("line", (line) => {
    let reply: Reply;
    try {
      reply = JSON.parse(line) as Reply;
    } catch {
      close(state, new Error("worker protocol error"));
      return;
    }
    const pending = state.pending;
    if (!pending) return;
    clearTimeout(pending.timer);
    state.pending = undefined;
    pending.resolve(reply);
  });
  child.once("error", () => close(state, new Error("worker unavailable")));
  child.stdin.on("error", () => close(state, new Error("worker unavailable")));
  child.stderr.resume();
  child.once("exit", () => close(state, new Error("worker exited")));
  const reply = await command(state, { action: "start", context, spend: true });
  if (!reply.ok) {
    close(state, new Error("worker rejected start"));
    throw new Error("worker rejected start");
  }
  return {
    session_id: state.id,
    models: { observer: "gpt-5.6-terra", writer: "gpt-5.6-terra" },
    limits: LIMITS,
  };
}

export async function batch(sessionId: string, playback_s: number, frames: unknown[]) {
  const state = globalThis.__liveWorker;
  if (!state || state.id !== sessionId) throw new Error("unknown session");
  if (state.pending) throw new Error("worker is busy");
  if (state.batches >= LIMITS.max_batches) {
    close(state, new Error("session limit reached"));
    throw new Error("session limit reached");
  }
  state.batches += 1;
  const reply = await command(state, { action: "batch", playback_s, frames });
  if (!reply.ok) {
    close(state);
    throw new Error("worker rejected batch");
  }
  const usage = reply.result?.usage as { cumulative?: { cost_usd?: number } } | undefined;
  state.cost = usage?.cumulative?.cost_usd ?? state.cost;
  if (state.cost >= LIMITS.max_cost_usd || reply.result?.stopped_reason) {
    close(state, new Error("cost limit reached"));
  }
  return reply.result;
}

export function end(sessionId: string) {
  const state = globalThis.__liveWorker;
  if (!state || state.id !== sessionId) throw new Error("unknown session");
  close(state);
  return { stopped: true };
}
