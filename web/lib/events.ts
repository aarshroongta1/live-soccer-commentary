/**
 * The SSE contract, in TypeScript.
 *
 * The Python bus flattens a pydantic model into the event payload and names
 * the SSE event after the topic, so the event name is the discriminant and
 * everything else is fields from `src/commentary/schemas.py`. Parsing happens
 * exactly once, here, on the way in: components see camelCase fields with
 * real types and never touch `JSON.parse` or an `unknown`.
 *
 * Every parser is defensive. A field the runtime has not learned to send yet
 * must degrade to a sane default rather than blank the panel.
 */

export const EVENT_NAMES = [
  "beat",
  "spoken",
  "preempted",
  "state",
  "board",
  "sighting",
  "caller",
  "analyst",
  "gate",
  "correction",
  "trigger",
  "cost",
  "status",
  "error",
  "phrased",
] as const;

export type EventName = (typeof EVENT_NAMES)[number];

/**
 * Topics the page subscribes to and deliberately drops.
 *
 * `phrased` carries the caller's own words beside the phrasing stage's
 * rewrite of them, so a trace can be read afterwards to say whether the
 * rewrite helped. Nothing on the page shows it — the line the viewer hears
 * arrives as a `beat` like any other — and it is listed here rather than
 * left out of `EVENT_NAMES` so that the contract stays the full set of
 * topics the runtime publishes.
 */
export const IGNORED_EVENTS = ["phrased"] as const;

export type IgnoredEvent = (typeof IGNORED_EVENTS)[number];

/** The topics that become a `StreamEvent`. */
export type RenderedEventName = Exclude<EventName, IgnoredEvent>;

export type Voice = "caller" | "analyst";

/** `Event` in schemas.py — what the caller thinks happened. */
export type MatchEvent =
  | "none"
  | "goal"
  | "shot"
  | "save"
  | "corner"
  | "free_kick"
  | "penalty"
  | "foul"
  | "offside"
  | "throw_in"
  | "card"
  | "substitution"
  | "kickoff"
  | "stoppage"
  | "build_up"
  | "pass"
  | "carry"
  | "interception"
  | "clearance"
  | "tackle"
  | "cross"
  | "switch";

/** Events the director will cut a line off for. */
export const BIG_EVENTS: ReadonlySet<MatchEvent> = new Set<MatchEvent>([
  "goal",
  "penalty",
  "card",
  "save",
]);

/** `Scene` in schemas.py — what the caller thinks it is looking at. */
export type Scene = "live_play" | "replay" | "close_up" | "crowd" | "stoppage" | "graphic";

export type Side = "home" | "away" | "unknown";

/**
 * `Trigger` in schemas.py — why the system considered speaking.
 *
 * There used to be a `whistle` and a `roar` here. Both detectors were
 * measured across every trace on disk and removed: the whistle fired once in
 * 58 runs, and the roar fired four times a minute whether or not anything
 * happened. See `docs/HANDOFF.md`, section 8.
 */
export type TriggerName = "camera_cut" | "board_change" | "silence_pressure" | "scheduled";

/** Fields every event carries, plus the two the client adds on arrival. */
interface Envelope<N extends EventName> {
  readonly name: N;
  /** Monotonic client-side id. Stable React key; the runtime has none. */
  readonly seq: number;
  /** Wall-clock arrival, for the "last event Ns ago" readout. */
  readonly at: number;
  /** Video timestamp the event is about, in seconds. */
  readonly ts: number;
}

/** The Beat model, shared by `beat`, `spoken` and `preempted`. */
interface BeatCore {
  readonly id: string;
  readonly voice: Voice;
  readonly text: string;
  readonly videoTs: number;
  /** Where the live edge was when the line was produced, on the same clock. */
  readonly liveTs: number;
  readonly event: MatchEvent;
  readonly urgency: number;
  readonly triggers: readonly TriggerName[];
  readonly preemptable: boolean;
}

export interface BeatEvent extends Envelope<"beat">, BeatCore {
  /** Set when the beat jumped the queue — a goal, a penalty, a card. */
  readonly urgent: boolean;
}

export interface SpokenEvent extends Envelope<"spoken">, BeatCore {
  readonly spoken: string;
  readonly seconds: number;
}

export interface PreemptedEvent extends Envelope<"preempted">, BeatCore {
  /** "cut" was killed mid-sentence; "stale" aged out before it was ever said. */
  readonly reason: "cut" | "stale";
  /** The part that made it out of the speaker before the cut. */
  readonly spoken: string;
  readonly seconds: number;
}

export interface StateEvent extends Envelope<"state"> {
  readonly home: string;
  readonly away: string;
  readonly homeScore: number;
  readonly awayScore: number;
  readonly clock: string | null;
  readonly period: number;
  readonly inReplay: boolean;
  readonly possession: Side;
}

export interface BoardEvent extends Envelope<"board"> {
  readonly bugVisible: boolean;
  readonly homeScore: number | null;
  readonly awayScore: number | null;
  readonly clock: string | null;
  readonly confidence: number;
}

/**
 * One shirt number or name the caller says it could read, and what the
 * roster made of it.
 *
 * `bound` is the whole point. A number the caller read off a shirt is a
 * claim about pixels; `as` is the player the team sheet turned it into, and
 * an unbound sighting is the caller reading something that names nobody.
 */
export interface Sighting {
  readonly number: number | null;
  readonly name: string | null;
  readonly side: Side;
  readonly bound: boolean;
  /** "26 Nahuel Molina" when the roster matched; null when it did not. */
  readonly as: string | null;
}

export interface CallerEvent extends Envelope<"caller"> {
  readonly scene: Scene;
  readonly event: MatchEvent;
  readonly side: Side;
  readonly team: string | null;
  /**
   * What the caller read off the picture. This used to be `names_read`, a
   * list of free text; the field on `CallerLine` is `sightings` and has been
   * since the second reader was removed from the schema.
   */
  readonly sightings: readonly Sighting[];
  readonly confidence: number;
  readonly speak: boolean;
  readonly line: string;
}

/**
 * The roster's verdict on everything one caller line read, published after
 * the line itself. Same list as `caller.sightings`, with the bind resolved.
 */
export interface SightingEvent extends Envelope<"sighting"> {
  readonly sightings: readonly Sighting[];
  readonly bound: number;
}

/** The statistician disagreeing with the screen, and the state giving way. */
export interface CorrectionEvent extends Envelope<"correction"> {
  readonly what: string;
  readonly event: MatchEvent;
}

export interface AnalystEvent extends Envelope<"analyst"> {
  readonly angle: string;
  readonly cites: readonly string[];
  readonly confidence: number;
  readonly speak: boolean;
  readonly line: string;
}

/** One reason from a gate verdict, shaped "tag: detail" on the wire. */
export interface GateReason {
  readonly tag: string;
  readonly detail: string;
}

export interface GateEvent extends Envelope<"gate"> {
  readonly passed: boolean;
  readonly reasons: readonly GateReason[];
  readonly line: string;
}

export interface TriggerEvent extends Envelope<"trigger"> {
  readonly triggers: readonly TriggerName[];
  readonly urgency: number;
  readonly reason: string;
  /** Absent on the wire in the early runtime; defaults to "triggers fired". */
  readonly shouldCall: boolean;
}

export interface CostAgent {
  readonly agent: string;
  readonly usd: number;
}

export interface CostEvent extends Envelope<"cost"> {
  readonly totalUsd: number;
  readonly perAgent: readonly CostAgent[];
}

/** `status` and `error` are free-form by contract; keep them printable. */
export interface NoticeEvent extends Envelope<"status" | "error"> {
  readonly level: "status" | "error";
  readonly text: string;
  readonly fields: readonly [string, string][];
}

export type StreamEvent =
  | BeatEvent
  | SpokenEvent
  | PreemptedEvent
  | StateEvent
  | BoardEvent
  | SightingEvent
  | CallerEvent
  | AnalystEvent
  | GateEvent
  | CorrectionEvent
  | TriggerEvent
  | CostEvent
  | NoticeEvent;

// -- coercion helpers -------------------------------------------------------

type Raw = Record<string, unknown>;

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function optNum(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function optStr(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function bool(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

const MATCH_EVENTS: readonly MatchEvent[] = [
  "none",
  "goal",
  "shot",
  "save",
  "corner",
  "free_kick",
  "penalty",
  "foul",
  "offside",
  "throw_in",
  "card",
  "substitution",
  "kickoff",
  "stoppage",
  "build_up",
  "pass",
  "carry",
  "interception",
  "clearance",
  "tackle",
  "cross",
  "switch",
];

const SCENES: readonly Scene[] = ["live_play", "replay", "close_up", "crowd", "stoppage", "graphic"];
const SIDES: readonly Side[] = ["home", "away", "unknown"];
const TRIGGER_NAMES: readonly TriggerName[] = [
  "camera_cut",
  "board_change",
  "silence_pressure",
  "scheduled",
];

/**
 * A sighting list, from either the caller's own form or the `sighting`
 * topic. The caller's copy has no `bound`/`as` yet — the bind happens after
 * the form is published — so an entry without them is simply unresolved.
 */
function sightings(value: unknown): Sighting[] {
  if (!Array.isArray(value)) return [];
  const parsed: Sighting[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const raw = item as Raw;
    parsed.push({
      number: optNum(raw.number),
      name: optStr(raw.name),
      side: oneOf(raw.side, SIDES, "unknown"),
      bound: bool(raw.bound),
      as: optStr(raw.as),
    });
  }
  return parsed;
}

function triggers(value: unknown): TriggerName[] {
  return strings(value).filter((item): item is TriggerName =>
    (TRIGGER_NAMES as readonly string[]).includes(item),
  );
}

function beatCore(raw: Raw): BeatCore {
  return {
    id: str(raw.id),
    voice: oneOf(raw.voice, ["caller", "analyst"] as const, "caller"),
    text: str(raw.text),
    videoTs: num(raw.video_ts, num(raw.ts)),
    liveTs: num(raw.live_ts),
    event: oneOf(raw.event, MATCH_EVENTS, "none"),
    urgency: num(raw.urgency),
    triggers: triggers(raw.triggers),
    preemptable: bool(raw.preemptable, true),
  };
}

/** Split a gate reason from its "tag: detail" wire shape. */
export function parseReason(reason: string): GateReason {
  const at = reason.indexOf(":");
  if (at < 0) return { tag: reason.trim(), detail: "" };
  return { tag: reason.slice(0, at).trim(), detail: reason.slice(at + 1).trim() };
}

/**
 * Cost arrives as `total_usd` plus whatever per-agent numbers the runtime has.
 * Both `{caller: 0.4}` and `{caller: {cost_usd: 0.4}}` are read, because the
 * shape of the second half of that payload is still settling.
 */
function costAgents(raw: Raw): CostAgent[] {
  const skip = new Set(["total_usd", "ts", "topic", "cost_usd"]);
  const agents: CostAgent[] = [];
  for (const [key, value] of Object.entries(raw)) {
    if (skip.has(key)) continue;
    if (typeof value === "number" && Number.isFinite(value)) {
      agents.push({ agent: key, usd: value });
    } else if (value && typeof value === "object" && !Array.isArray(value)) {
      const nested = value as Raw;
      const usd = optNum(nested.cost_usd) ?? optNum(nested.usd);
      if (usd !== null) agents.push({ agent: key, usd });
    }
  }
  return agents.sort((a, b) => b.usd - a.usd);
}

function noticeText(raw: Raw): string {
  for (const key of ["message", "detail", "text", "status", "value", "error"]) {
    const found = optStr(raw[key]);
    if (found) return found;
  }
  return "";
}

function noticeFields(raw: Raw): [string, string][] {
  const skip = new Set(["ts", "topic", "message", "detail", "text", "status", "value", "error"]);
  return Object.entries(raw)
    .filter(([key]) => !skip.has(key))
    .map(([key, value]) => [key, typeof value === "string" ? value : JSON.stringify(value)] as [string, string]);
}

let sequence = 0;

/**
 * Turn one SSE frame into a typed event, or null if the payload is unusable.
 * The event *name* is authoritative — the payload's own `topic` field is
 * advisory, since a proxy could in principle reorder or rewrite it.
 */
export function parseEvent(name: string, data: string): StreamEvent | null {
  if (!(EVENT_NAMES as readonly string[]).includes(name)) return null;
  if ((IGNORED_EVENTS as readonly string[]).includes(name)) return null;
  let raw: Raw;
  try {
    const parsed: unknown = JSON.parse(data);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    raw = parsed as Raw;
  } catch {
    return null;
  }
  return buildEvent(name as RenderedEventName, raw);
}

/** The parser proper, exposed so the fixture replays through the same path. */
export function buildEvent(name: RenderedEventName, raw: Raw): StreamEvent {
  const envelope = { seq: sequence++, at: Date.now(), ts: num(raw.ts) };

  switch (name) {
    case "beat":
      return { name, ...envelope, ...beatCore(raw), urgent: bool(raw.urgent) };
    case "spoken":
      return {
        name,
        ...envelope,
        ...beatCore(raw),
        spoken: str(raw.spoken, str(raw.text)),
        seconds: num(raw.seconds),
      };
    case "preempted":
      return {
        name,
        ...envelope,
        ...beatCore(raw),
        reason: raw.reason === "stale" ? "stale" : "cut",
        spoken: str(raw.spoken),
        seconds: num(raw.seconds),
      };
    case "state":
      return {
        name,
        ...envelope,
        home: str(raw.home, "Home"),
        away: str(raw.away, "Away"),
        homeScore: num(raw.home_score),
        awayScore: num(raw.away_score),
        clock: optStr(raw.clock),
        period: num(raw.period, 1),
        inReplay: bool(raw.in_replay),
        possession: oneOf(raw.possession, SIDES, "unknown"),
      };
    case "board":
      return {
        name,
        ...envelope,
        bugVisible: bool(raw.bug_visible),
        homeScore: optNum(raw.home_score),
        awayScore: optNum(raw.away_score),
        clock: optStr(raw.clock),
        confidence: num(raw.confidence),
      };
    case "caller":
      return {
        name,
        ...envelope,
        scene: oneOf(raw.scene, SCENES, "live_play"),
        event: oneOf(raw.event, MATCH_EVENTS, "none"),
        side: oneOf(raw.side, SIDES, "unknown"),
        team: optStr(raw.team),
        sightings: sightings(raw.sightings),
        confidence: num(raw.confidence),
        speak: bool(raw.speak),
        line: str(raw.line),
      };
    case "sighting": {
      const seen = sightings(raw.sightings);
      return {
        name,
        ...envelope,
        sightings: seen,
        bound: seen.filter((item) => item.bound).length,
      };
    }
    case "correction":
      return {
        name,
        ...envelope,
        what: str(raw.what),
        event: oneOf(raw.event, MATCH_EVENTS, "none"),
      };
    case "analyst":
      return {
        name,
        ...envelope,
        angle: str(raw.angle),
        cites: strings(raw.cites),
        confidence: num(raw.confidence),
        speak: bool(raw.speak),
        line: str(raw.line),
      };
    case "gate":
      return {
        name,
        ...envelope,
        passed: bool(raw.passed),
        reasons: strings(raw.reasons).map(parseReason),
        line: str(raw.line),
      };
    case "trigger": {
      const fired = triggers(raw.triggers);
      return {
        name,
        ...envelope,
        triggers: fired,
        urgency: num(raw.urgency),
        reason: str(raw.reason),
        shouldCall: bool(raw.should_call, fired.length > 0),
      };
    }
    case "cost":
      return {
        name,
        ...envelope,
        totalUsd: num(raw.total_usd, num(raw.cost_usd)),
        perAgent: costAgents(raw),
      };
    case "status":
    case "error":
      return {
        name,
        ...envelope,
        level: name,
        text: noticeText(raw),
        fields: noticeFields(raw),
      };
  }
}
