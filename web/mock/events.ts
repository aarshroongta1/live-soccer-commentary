/**
 * One realistic minute of a match, in the wire shape the Python bus emits.
 *
 * This is the fixture the UI is developed against: it runs through the same
 * `buildEvent` parser as the live stream, so a field that would break the page
 * at 88 minutes on a Tuesday breaks it here first. It covers the six things
 * the agent panel exists to show — a routine build-up, a shot, a shirt read
 * that binds to nobody, a save whose line the gate rejects, a deliberate
 * silence, and a goal that cuts the analyst off mid-sentence.
 *
 * `after` is milliseconds to wait before emitting, relative to the previous
 * frame. `ts` inside each payload is video time, which advances in step.
 */

import type { EventName } from "@/lib/events";

export interface FixtureFrame {
  readonly after: number;
  readonly name: EventName;
  readonly data: Record<string, unknown>;
}

const HOME = "Bayern";
const AWAY = "Inter";

/** Beat payloads repeat nine fields; this keeps the fixture readable. */
function beat(
  id: string,
  ts: number,
  voice: "caller" | "analyst",
  text: string,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    topic: "beat",
    ts,
    id,
    voice,
    text,
    video_ts: ts,
    created_ts: ts,
    // The live edge when the line was produced: eight seconds of buffer plus
    // the model's own latency.
    live_ts: ts + 9.2,
    event: "none",
    urgency: 0.3,
    triggers: ["scheduled"],
    preemptable: true,
    ...extra,
  };
}

export const FIXTURE: readonly FixtureFrame[] = [
  {
    after: 0,
    name: "status",
    data: { ts: 3124.0, message: "runtime attached · caller sonnet-5 · board haiku-4.5 · delay 8.0s" },
  },
  {
    after: 40,
    name: "state",
    data: {
      ts: 3124.0,
      home: HOME,
      away: AWAY,
      home_score: 1,
      away_score: 0,
      clock: "52:04",
      period: 2,
      in_replay: false,
      possession: "home",
    },
  },
  {
    after: 40,
    name: "cost",
    data: { ts: 3124.0, total_usd: 4.2116, caller: 3.0182, board: 0.4093, analyst: 0.6789, researcher: 0.1052 },
  },
  {
    after: 160,
    name: "board",
    data: { ts: 3124.2, bug_visible: true, home_score: 1, away_score: 0, clock: "52:04", confidence: 0.96 },
  },

  // -- a routine build-up ---------------------------------------------------
  {
    after: 900,
    name: "trigger",
    data: { ts: 3125.1, triggers: ["scheduled"], urgency: 0.24, reason: "4.1s since last line", should_call: true },
  },
  {
    after: 350,
    name: "caller",
    data: {
      ts: 3125.4,
      scene: "live_play",
      event: "build_up",
      side: "home",
      team: HOME,
      sightings: [{ number: 6, name: "Kimmich", side: "home", bound: true, as: "6 Joshua Kimmich" }],
      confidence: 0.74,
      speak: true,
      line: "Kimmich takes it off the back line, Bayern in no hurry here.",
    },
  },
  {
    after: 40,
    name: "sighting",
    data: {
      ts: 3125.4,
      sightings: [{ number: 6, name: "Kimmich", side: "home", bound: true, as: "6 Joshua Kimmich" }],
    },
  },
  {
    after: 120,
    name: "gate",
    data: {
      ts: 3125.4,
      passed: true,
      reasons: ["roster: Kimmich on Bayern team sheet", "score: no scoreline claimed"],
      line: "Kimmich takes it off the back line, Bayern in no hurry here.",
    },
  },
  {
    after: 60,
    name: "beat",
    data: beat("b-4471", 3125.4, "caller", "Kimmich takes it off the back line, Bayern in no hurry here.", {
      event: "build_up",
      urgency: 0.24,
    }),
  },
  {
    after: 300,
    name: "spoken",
    data: beat("b-4471", 3125.4, "caller", "Kimmich takes it off the back line, Bayern in no hurry here.", {
      topic: "spoken",
      event: "build_up",
      urgency: 0.24,
      spoken: "Kimmich takes it off the back line, Bayern in no hurry here.",
      seconds: 3.4,
    }),
  },

  // -- a camera cut the caller declines to narrate --------------------------
  {
    after: 2400,
    name: "board",
    data: { ts: 3128.2, bug_visible: true, home_score: 1, away_score: 0, clock: "52:08", confidence: 0.94 },
  },
  {
    after: 700,
    name: "trigger",
    data: {
      ts: 3128.9,
      triggers: ["camera_cut"],
      urgency: 0.18,
      reason: "cut to bench, nothing on the pitch to call",
      should_call: true,
    },
  },
  {
    after: 300,
    name: "caller",
    data: {
      ts: 3129.2,
      scene: "close_up",
      event: "none",
      side: "unknown",
      team: null,
      sightings: [],
      confidence: 0.55,
      speak: false,
      line: "",
    },
  },

  // -- a shot ---------------------------------------------------------------
  {
    after: 3600,
    name: "trigger",
    data: { ts: 3132.8, triggers: ["camera_cut"], urgency: 0.72, reason: "cut to a tight shot, 0.8s", should_call: true },
  },
  {
    after: 280,
    name: "caller",
    data: {
      ts: 3133.1,
      scene: "live_play",
      event: "shot",
      side: "home",
      team: HOME,
      sightings: [{ number: 10, name: "Musiala", side: "home", bound: true, as: "10 Jamal Musiala" }],
      confidence: 0.81,
      speak: true,
      line: "Musiala cuts inside and goes for the far corner!",
    },
  },
  {
    after: 40,
    name: "sighting",
    data: {
      ts: 3133.1,
      sightings: [{ number: 10, name: "Musiala", side: "home", bound: true, as: "10 Jamal Musiala" }],
    },
  },
  {
    after: 110,
    name: "gate",
    data: {
      ts: 3133.1,
      passed: true,
      reasons: ["roster: Musiala on Bayern team sheet", "event: shot needs no board change"],
      line: "Musiala cuts inside and goes for the far corner!",
    },
  },
  {
    after: 50,
    name: "beat",
    data: beat("b-4472", 3133.1, "caller", "Musiala cuts inside and goes for the far corner!", {
      event: "shot",
      urgency: 0.72,
      triggers: ["camera_cut"],
    }),
  },
  {
    after: 260,
    name: "spoken",
    data: beat("b-4472", 3133.1, "caller", "Musiala cuts inside and goes for the far corner!", {
      topic: "spoken",
      event: "shot",
      urgency: 0.72,
      triggers: ["camera_cut"],
      spoken: "Musiala cuts inside and goes for the far corner!",
      seconds: 2.6,
    }),
  },

  // -- the save, and a line the gate refuses to let out ---------------------
  {
    after: 1500,
    name: "trigger",
    data: { ts: 3134.9, triggers: ["camera_cut"], urgency: 0.64, reason: "cut to the goalmouth", should_call: true },
  },
  {
    after: 260,
    name: "caller",
    data: {
      ts: 3135.2,
      scene: "live_play",
      event: "save",
      side: "away",
      team: AWAY,
      sightings: [{ number: null, name: "Sommer", side: "away", bound: false, as: null }],
      confidence: 0.42,
      speak: true,
      line: "Sommer gets a strong hand to it and turns it behind for the corner.",
    },
  },
  {
    after: 40,
    name: "sighting",
    data: {
      ts: 3135.2,
      sightings: [{ number: null, name: "Sommer", side: "away", bound: false, as: null }],
    },
  },
  {
    after: 90,
    name: "gate",
    data: {
      ts: 3135.2,
      passed: false,
      reasons: [
        "roster: 'Sommer' is on neither team sheet",
        "confidence: 0.42 below the 0.60 floor for a named save",
      ],
      line: "Sommer gets a strong hand to it and turns it behind for the corner.",
    },
  },
  {
    after: 420,
    name: "caller",
    data: {
      ts: 3135.6,
      scene: "live_play",
      event: "save",
      side: "away",
      team: AWAY,
      sightings: [],
      confidence: 0.69,
      speak: true,
      line: "The keeper gets a strong hand to it, behind for a corner.",
    },
  },
  {
    after: 90,
    name: "gate",
    data: {
      ts: 3135.6,
      passed: true,
      reasons: ["roster: no name claimed", "event: save needs no board change"],
      line: "The keeper gets a strong hand to it, behind for a corner.",
    },
  },
  {
    after: 50,
    name: "beat",
    data: beat("b-4473", 3135.6, "caller", "The keeper gets a strong hand to it, behind for a corner.", {
      event: "save",
      urgency: 0.64,
      triggers: ["camera_cut"],
    }),
  },
  {
    after: 240,
    name: "spoken",
    data: beat("b-4473", 3135.6, "caller", "The keeper gets a strong hand to it, behind for a corner.", {
      topic: "spoken",
      event: "save",
      urgency: 0.64,
      triggers: ["camera_cut"],
      spoken: "The keeper gets a strong hand to it, behind for a corner.",
      seconds: 3.1,
    }),
  },

  // -- a replay, which is not narrated as live ------------------------------
  {
    after: 1800,
    name: "board",
    data: { ts: 3137.6, bug_visible: false, home_score: null, away_score: null, clock: null, confidence: 0.88 },
  },
  {
    after: 80,
    name: "state",
    data: {
      ts: 3137.6,
      home: HOME,
      away: AWAY,
      home_score: 1,
      away_score: 0,
      clock: "52:17",
      period: 2,
      in_replay: true,
      possession: "unknown",
    },
  },
  {
    after: 900,
    name: "caller",
    data: {
      ts: 3138.5,
      scene: "replay",
      event: "save",
      side: "away",
      team: AWAY,
      sightings: [],
      confidence: 0.77,
      speak: false,
      line: "",
    },
  },
  {
    after: 2600,
    name: "board",
    data: { ts: 3141.1, bug_visible: true, home_score: 1, away_score: 0, clock: "52:21", confidence: 0.95 },
  },
  {
    after: 80,
    name: "state",
    data: {
      ts: 3141.1,
      home: HOME,
      away: AWAY,
      home_score: 1,
      away_score: 0,
      clock: "52:21",
      period: 2,
      in_replay: false,
      possession: "home",
    },
  },

  // -- a lull, so the analyst gets the floor --------------------------------
  {
    after: 2200,
    name: "trigger",
    data: {
      ts: 3143.3,
      triggers: ["silence_pressure"],
      urgency: 0.31,
      reason: "7.7s of silence, play settled",
      should_call: true,
    },
  },
  {
    after: 900,
    name: "analyst",
    data: {
      ts: 3144.2,
      angle: "tactics",
      cites: ["Inter have dropped to a back five since the hour", "Bayern 63% possession"],
      confidence: 0.8,
      speak: true,
      line: "Inter have gone to a back five since the hour mark, and it has handed Bayern the ball in exactly the areas they want it.",
    },
  },
  {
    after: 120,
    name: "beat",
    data: beat(
      "b-4474",
      3144.2,
      "analyst",
      "Inter have gone to a back five since the hour mark, and it has handed Bayern the ball in exactly the areas they want it.",
      { urgency: 0.31, triggers: ["silence_pressure"] },
    ),
  },

  // -- the goal, which takes the analyst's microphone away ------------------
  {
    after: 2100,
    name: "trigger",
    data: {
      ts: 3146.3,
      triggers: ["board_change"],
      urgency: 0.98,
      reason: "board 1-0 to 2-0, third agreeing read",
      should_call: true,
    },
  },
  {
    after: 180,
    name: "board",
    data: { ts: 3146.5, bug_visible: true, home_score: 2, away_score: 0, clock: "52:26", confidence: 0.93 },
  },
  {
    after: 140,
    name: "caller",
    data: {
      ts: 3146.6,
      scene: "live_play",
      event: "goal",
      side: "home",
      team: HOME,
      sightings: [{ number: 10, name: "Musiala", side: "home", bound: true, as: "10 Jamal Musiala" }],
      confidence: 0.93,
      speak: true,
      line: "Musiala! Two-nil Bayern, and Inter are gone.",
    },
  },
  {
    after: 40,
    name: "sighting",
    data: {
      ts: 3146.6,
      sightings: [{ number: 10, name: "Musiala", side: "home", bound: true, as: "10 Jamal Musiala" }],
    },
  },
  {
    after: 70,
    name: "gate",
    data: {
      ts: 3146.6,
      passed: true,
      reasons: [
        "roster: Musiala on Bayern team sheet",
        "score: 2-0 matches the board",
        "event: goal confirmed by board change at 3146.5",
      ],
      line: "Musiala! Two-nil Bayern, and Inter are gone.",
    },
  },
  {
    after: 40,
    name: "beat",
    data: beat("b-4475", 3146.6, "caller", "Musiala! Two-nil Bayern, and Inter are gone.", {
      event: "goal",
      urgency: 0.98,
      triggers: ["board_change"],
      preemptable: false,
      urgent: true,
    }),
  },
  {
    after: 30,
    name: "preempted",
    data: beat(
      "b-4474",
      3144.2,
      "analyst",
      "Inter have gone to a back five since the hour mark, and it has handed Bayern the ball in exactly the areas they want it.",
      {
        topic: "preempted",
        urgency: 0.31,
        triggers: ["silence_pressure"],
        reason: "cut",
        spoken: "Inter have gone to a back five since the hour mark, and it has handed Bayern the",
        seconds: 2.4,
      },
    ),
  },
  {
    after: 260,
    name: "spoken",
    data: beat("b-4475", 3146.6, "caller", "Musiala! Two-nil Bayern, and Inter are gone.", {
      topic: "spoken",
      event: "goal",
      urgency: 0.98,
      triggers: ["board_change"],
      preemptable: false,
      spoken: "Musiala! Two-nil Bayern, and Inter are gone.",
      seconds: 2.9,
    }),
  },
  {
    after: 120,
    name: "state",
    data: {
      ts: 3146.9,
      home: HOME,
      away: AWAY,
      home_score: 2,
      away_score: 0,
      clock: "52:26",
      period: 2,
      in_replay: false,
      possession: "away",
    },
  },

  // -- and one more rejection, this time for a scoreline it invented --------
  {
    after: 2600,
    name: "caller",
    data: {
      ts: 3149.5,
      scene: "live_play",
      event: "none",
      side: "home",
      team: HOME,
      sightings: [],
      confidence: 0.58,
      speak: true,
      line: "Three-nil, and this is a rout now.",
    },
  },
  {
    after: 80,
    name: "gate",
    data: {
      ts: 3149.5,
      passed: false,
      reasons: ["score: line claims 3-0, board reads 2-0"],
      line: "Three-nil, and this is a rout now.",
    },
  },
  {
    after: 700,
    name: "cost",
    data: { ts: 3150.2, total_usd: 4.3391, caller: 3.1204, board: 0.4127, analyst: 0.6998, researcher: 0.1052 },
  },
  {
    after: 900,
    name: "preempted",
    data: beat("b-4476", 3139.8, "caller", "Still Bayern, still patient through the middle third.", {
      topic: "preempted",
      urgency: 0.2,
      reason: "stale",
      spoken: "",
      seconds: 0,
    }),
  },
  {
    after: 1200,
    name: "board",
    data: { ts: 3152.3, bug_visible: true, home_score: 2, away_score: 0, clock: "52:32", confidence: 0.97 },
  },
];

/** How long one pass through the fixture takes, in milliseconds. */
export const FIXTURE_DURATION_MS = FIXTURE.reduce((total, frame) => total + frame.after, 0);
