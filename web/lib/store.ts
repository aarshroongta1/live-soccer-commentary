/**
 * What the page remembers about the match so far.
 *
 * A 90 minute match is a long stream, so nothing here grows without a bound:
 * the transcript keeps the last few hundred lines and each panel keeps a short
 * tail. Everything else is last-write-wins, because a scoreboard and a cost
 * ledger have no history worth holding in a browser tab.
 */

import type {
  BoardEvent,
  CallerEvent,
  CostEvent,
  GateEvent,
  NoticeEvent,
  PreemptedEvent,
  SpokenEvent,
  StateEvent,
  StreamEvent,
  TriggerEvent,
} from "@/lib/events";

/** Lines the transcript shows: what was said, and what was killed saying it. */
export type FeedItem = SpokenEvent | PreemptedEvent;

export const FEED_CAP = 400;
export const PANEL_CAP = 60;

export interface Counts {
  readonly beats: number;
  readonly spoken: number;
  readonly cut: number;
  readonly stale: number;
  readonly gatePassed: number;
  readonly gateRejected: number;
}

export interface StreamStore {
  readonly state: StateEvent | null;
  readonly board: BoardEvent | null;
  readonly caller: CallerEvent | null;
  readonly trigger: TriggerEvent | null;
  readonly cost: CostEvent | null;
  readonly notice: NoticeEvent | null;
  /** Oldest first — a transcript reads downward. */
  readonly feed: readonly FeedItem[];
  /** Newest first — a panel of recent rejections reads downward from now. */
  readonly rejected: readonly GateEvent[];
  readonly cuts: readonly PreemptedEvent[];
  readonly counts: Counts;
  /**
   * How far the last spoken line trailed the live edge: buffer depth plus
   * however long the model took. Null until a beat carries a live edge.
   */
  readonly lastLagS: number | null;
  /** Arrival time of the most recent event of any kind. */
  readonly lastEventAt: number | null;
}

export const EMPTY_STORE: StreamStore = {
  state: null,
  board: null,
  caller: null,
  trigger: null,
  cost: null,
  notice: null,
  feed: [],
  rejected: [],
  cuts: [],
  counts: { beats: 0, spoken: 0, cut: 0, stale: 0, gatePassed: 0, gateRejected: 0 },
  lastLagS: null,
  lastEventAt: null,
};

/** A mutable working copy, so a batch of events costs one object allocation. */
interface Draft {
  state: StateEvent | null;
  board: BoardEvent | null;
  caller: CallerEvent | null;
  trigger: TriggerEvent | null;
  cost: CostEvent | null;
  notice: NoticeEvent | null;
  feed: FeedItem[];
  rejected: GateEvent[];
  cuts: PreemptedEvent[];
  counts: Counts;
  lastLagS: number | null;
  lastEventAt: number | null;
  touched: boolean;
}

function pushCapped<T>(list: T[], item: T, cap: number): void {
  list.push(item);
  if (list.length > cap) list.splice(0, list.length - cap);
}

function unshiftCapped<T>(list: T[], item: T, cap: number): void {
  list.unshift(item);
  if (list.length > cap) list.length = cap;
}

function apply(draft: Draft, event: StreamEvent): void {
  draft.lastEventAt = event.at;
  draft.touched = true;

  switch (event.name) {
    case "beat":
      draft.counts = { ...draft.counts, beats: draft.counts.beats + 1 };
      break;
    case "spoken":
      draft.counts = { ...draft.counts, spoken: draft.counts.spoken + 1 };
      if (event.liveTs > 0) draft.lastLagS = event.liveTs - event.videoTs;
      pushCapped(draft.feed, event, FEED_CAP);
      break;
    case "preempted":
      draft.counts =
        event.reason === "cut"
          ? { ...draft.counts, cut: draft.counts.cut + 1 }
          : { ...draft.counts, stale: draft.counts.stale + 1 };
      unshiftCapped(draft.cuts, event, PANEL_CAP);
      // A line cut mid-sentence still made it to air in part, so it belongs in
      // the transcript. One that aged out was never spoken, and does not.
      if (event.reason === "cut" && event.spoken.length > 0) {
        pushCapped(draft.feed, event, FEED_CAP);
      }
      break;
    case "state":
      draft.state = event;
      break;
    case "board":
      draft.board = event;
      break;
    case "caller":
      draft.caller = event;
      break;
    case "trigger":
      draft.trigger = event;
      break;
    case "gate":
      if (event.passed) {
        draft.counts = { ...draft.counts, gatePassed: draft.counts.gatePassed + 1 };
      } else {
        draft.counts = { ...draft.counts, gateRejected: draft.counts.gateRejected + 1 };
        unshiftCapped(draft.rejected, event, PANEL_CAP);
      }
      break;
    case "cost":
      draft.cost = event;
      break;
    case "status":
    case "error":
      draft.notice = event;
      break;
    case "analyst":
      // The analyst's structured form becomes a beat of its own the moment the
      // director accepts it; nothing extra to hold here.
      break;
  }
}

/** Fold a batch of events into the store, allocating a new one only if needed. */
export function reduceEvents(store: StreamStore, events: readonly StreamEvent[]): StreamStore {
  if (events.length === 0) return store;

  const draft: Draft = {
    state: store.state,
    board: store.board,
    caller: store.caller,
    trigger: store.trigger,
    cost: store.cost,
    notice: store.notice,
    feed: [...store.feed],
    rejected: [...store.rejected],
    cuts: [...store.cuts],
    counts: store.counts,
    lastLagS: store.lastLagS,
    lastEventAt: store.lastEventAt,
    touched: false,
  };

  for (const event of events) apply(draft, event);
  if (!draft.touched) return store;

  return {
    state: draft.state,
    board: draft.board,
    caller: draft.caller,
    trigger: draft.trigger,
    cost: draft.cost,
    notice: draft.notice,
    feed: draft.feed,
    rejected: draft.rejected,
    cuts: draft.cuts,
    counts: draft.counts,
    lastLagS: draft.lastLagS,
    lastEventAt: draft.lastEventAt,
  };
}
