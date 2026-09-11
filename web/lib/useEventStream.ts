"use client";

import { useEffect, useRef, useState } from "react";

import { EVENT_NAMES, buildEvent, parseEvent, type StreamEvent } from "@/lib/events";
import { EMPTY_STORE, reduceEvents, type StreamStore } from "@/lib/store";
import { FIXTURE } from "@/mock/events";

export type StreamSource = "live" | "fixture";

export type Connection = "connecting" | "open" | "reconnecting" | "replaying";

export interface StreamHandle {
  readonly store: StreamStore;
  readonly connection: Connection;
  /** Consecutive failed connection attempts; zero once the stream is open. */
  readonly attempts: number;
  readonly source: StreamSource;
}

/** Events are folded into the store on a tick, not per message. */
const FLUSH_MS = 100;
/** Pause between passes through the fixture, so a loop is legible as a loop. */
const LOOP_GAP_MS = 4000;
const BACKOFF_BASE_MS = 750;
const BACKOFF_MAX_MS = 15_000;

function backoff(attempt: number): number {
  return Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** attempt) + Math.random() * 250;
}

/**
 * One EventSource for the whole page.
 *
 * Every panel reads from the single store this returns; nothing opens a stream
 * of its own, because a second subscriber costs the runtime a second bounded
 * queue and gives the viewer a second, differently-lagged version of the match.
 *
 * Messages are buffered and folded in on a 100ms tick. A busy passage emits
 * board reads, triggers and beats faster than React should re-render, and the
 * viewer cannot tell a 100ms batch from an instant one.
 */
export function useEventStream(source: StreamSource): StreamHandle {
  const [store, setStore] = useState<StreamStore>(EMPTY_STORE);
  const [liveConnection, setLiveConnection] = useState<Connection>("connecting");
  const [attempts, setAttempts] = useState(0);
  const [shownSource, setShownSource] = useState(source);
  const queue = useRef<StreamEvent[]>([]);

  // Switching source starts a new match as far as the page is concerned. This
  // is the one place React sanctions setting state during render: derived
  // state that has to be thrown away when a prop changes.
  if (shownSource !== source) {
    setShownSource(source);
    setStore(EMPTY_STORE);
    setLiveConnection("connecting");
    setAttempts(0);
  }

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (queue.current.length === 0) return;
      const batch = queue.current;
      queue.current = [];
      setStore((current) => reduceEvents(current, batch));
    }, FLUSH_MS);
    return () => window.clearInterval(timer);
  }, []);

  // Events buffered from the previous source must not land in the new match.
  useEffect(() => {
    queue.current = [];
    return () => {
      queue.current = [];
    };
  }, [source]);

  useEffect(() => {
    if (source !== "live") return;

    let cancelled = false;
    let stream: EventSource | null = null;
    let retry: number | undefined;

    /** The snapshot fills the scoreboard before the first event arrives. */
    const seed = async () => {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (!response.ok || cancelled) return;
        const body: unknown = await response.json();
        if (cancelled || !body || typeof body !== "object") return;
        const { state, status, usage } = body as Record<string, unknown>;
        if (state && typeof state === "object") {
          queue.current.push(buildEvent("state", { ts: 0, ...(state as Record<string, unknown>) }));
        }
        if (usage && typeof usage === "object") {
          const cost = (usage as Record<string, unknown>).cost_usd;
          queue.current.push(buildEvent("cost", { ts: 0, total_usd: cost }));
        }
        if (status && typeof status === "object") {
          queue.current.push(buildEvent("status", { ts: 0, ...(status as Record<string, unknown>) }));
        }
      } catch {
        // The stream is the source of truth; a missing snapshot only means the
        // page starts blank rather than pre-filled.
      }
    };

    const connect = (attempt: number) => {
      const opened = new EventSource("/api/stream");
      stream = opened;

      opened.onopen = () => {
        if (cancelled) return;
        setLiveConnection("open");
        setAttempts(0);
        void seed();
      };

      for (const name of EVENT_NAMES) {
        opened.addEventListener(name, (raw: Event) => {
          const message = raw as MessageEvent<string>;
          const event = parseEvent(name, message.data);
          if (event) queue.current.push(event);
        });
      }

      opened.onerror = () => {
        if (cancelled) return;
        // EventSource would retry on its own schedule; closing it here means
        // the backoff below is the only retry policy, and it is visible.
        opened.close();
        const next = attempt + 1;
        setAttempts(next);
        setLiveConnection("reconnecting");
        retry = window.setTimeout(() => connect(next), backoff(attempt));
      };
    };

    connect(0);

    return () => {
      cancelled = true;
      stream?.close();
      if (retry !== undefined) window.clearTimeout(retry);
    };
  }, [source]);

  useEffect(() => {
    if (source !== "fixture") return;

    let index = 0;
    let timer: number;

    const tick = () => {
      const frame = FIXTURE[index];
      queue.current.push(buildEvent(frame.name, frame.data));
      index = (index + 1) % FIXTURE.length;
      timer = window.setTimeout(tick, index === 0 ? LOOP_GAP_MS : FIXTURE[index].after);
    };

    timer = window.setTimeout(tick, FIXTURE[0].after);
    return () => window.clearTimeout(timer);
  }, [source]);

  // The fixture is never "connecting"; it is simply playing.
  const connection: Connection = source === "fixture" ? "replaying" : liveConnection;

  return { store, connection, attempts, source };
}
