"""Runtime settings. Everything tunable in one place.

Every threshold in this file is a claim about what makes commentary sound
right, and every one of them is measured in the eval. They live here so a
sweep changes one object rather than six modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Models (see the claude-api skill for current ids).
CALLER_MODEL = os.getenv("CALLER_MODEL", "claude-sonnet-5")
BOARD_MODEL = os.getenv("BOARD_MODEL", "claude-haiku-4-5")
ANALYST_MODEL = os.getenv("ANALYST_MODEL", "claude-opus-5")
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "claude-opus-5")
#: The phrasing stage. Haiku by default: it is a rewrite of a form into six
#: words, with the examples that teach the register sitting in a cached
#: system prompt, so the per-line cost is a fraction of a cent. Set it to
#: ``off`` and the stage does not exist — the caller's own line goes to the
#: gate, exactly as before there was a phraser.
PHRASER_MODEL = os.getenv("PHRASER_MODEL", "claude-haiku-4-5")
#: The colour seat. Haiku for the same reason the phraser is: the seat is
#: given the lead's last lines, the forms since its own last turn and the
#: pack notes, and asked for three short utterances. The rules and the
#: forty real colour utterances that set the register sit in a cached system
#: prefix, so a turn costs a fraction of a cent. ``off`` removes the seat.
COLOUR_MODEL = os.getenv("COLOUR_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5")


@dataclass(frozen=True)
class CaptureConfig:
    """Screen capture and the delay buffer that sits behind it."""

    #: Prefer the name form, e.g. "Capture screen 0", over an index like
    #: "1:0": avfoundation device indices shift when hardware is plugged in
    #: (the screen was index 1, then became 3 with an iPhone attached), while
    #: the name is stable across those changes.
    device: str = os.getenv("AVFOUNDATION_DEVICE", "Capture screen 0")
    width: int = 1280
    height: int = 720
    fps: int = 15
    #: How far the narration cursor trails the live edge. The headline
    #: experiment sweeps this at 0, 2, 4, 8 seconds.
    delay_s: float = float(os.getenv("DELAY_S", "8.0"))
    #: Seconds of frames kept behind the cursor, for lookback in prompts.
    history_s: float = 6.0
    #: How far behind the narration cursor the viewer's picture is held.
    #:
    #: A caller line is stamped with the cursor at the moment the call
    #: *starts* and reaches the viewer when the call returns, so without this
    #: every line lands after the moment it describes has gone past on screen.
    #: Measured on the 60 s screen run —
    #: ``runs/screen/dimaria/screen-20260913-222906.jsonl``, ``live_ts -
    #: video_ts - delay_s`` over its ``beat`` rows — the round trip is a
    #: median 3.4 s, range 1.9 to 5.6 s, and the median was the same on Opus
    #: over file runs. Holding the picture back by that much puts the line
    #: and its moment back together.
    #:
    #: The buffer is what pays for it: the frame comes from ``history_s``
    #: behind the cursor, so the offset can never exceed it. Shrinking the
    #: delay instead is not an option — the gate's goal window and the
    #: caller's lookahead are both spent out of it.
    present_offset_s: float = float(os.getenv("PRESENT_OFFSET_S", "3.5"))

    def __post_init__(self) -> None:
        if self.present_offset_s < 0:
            raise ValueError("present_offset_s must be >= 0")
        if self.present_offset_s > self.history_s:
            raise ValueError(
                f"present_offset_s ({self.present_offset_s:g}s) is more than the buffer keeps "
                f"behind the cursor (history_s {self.history_s:g}s): the viewer's frame would "
                f"already have been evicted. Raise history_s or lower PRESENT_OFFSET_S."
            )


@dataclass(frozen=True)
class BoardConfig:
    """Reading the score bug.

    The crop is a fraction of the frame, not pixels, so one preset survives a
    change of resolution. Broadcasters put the bug in different places; the
    presets are named after where it sits, not after the broadcaster.
    """

    #: (x0, y0, x1, y1) as fractions of width and height.
    crop: tuple[float, float, float, float] = (0.0, 0.0, 0.42, 0.16)
    interval_s: float = 2.0
    #: A score only changes after this many agreeing reads. Stops one bad read
    #: from inventing a goal.
    confirmations: int = 3
    min_confidence: float = 0.6


@dataclass(frozen=True)
class CallerConfig:
    """The play-by-play voice."""

    frames_at_cursor: int = 4
    cursor_spacing_s: float = 1.0
    frames_lookahead: int = 2
    #: The longest the voice ever waits between two lines on the rate cap
    #: alone. A full sentence takes about this long to say, so after one the
    #: next line lands as the last one finishes.
    min_gap_s: float = 4.0
    #: The shortest that wait is ever allowed to get. Real commentary calls
    #: build-up in fragments — "De Paul." "Messi, Álvarez." — and a two-word
    #: line held for four seconds is three seconds of dead air. Under 1.5s
    #: two lines tread on each other.
    #:
    #: This used to cite "a median 2.4s apart", which was the wrong unit:
    #: 2.44s is the gap between YouTube *caption segments*, of which there
    #: are 1.75 to an utterance. Utterance to utterance the real median is
    #: 4.3s pooled across club football and 4.6s at the 2022 final, and more
    #: than half of all gaps run over four seconds (study section 2.1). The
    #: floor is not changed here — what it is a floor on is a rate decision
    #: and study Gap 3 asks for two rates, not a different one — but the
    #: number it was justified by was measuring something else.
    min_gap_floor_s: float = 1.5
    #: Breath. Added to however long the last line took, so the gap is a
    #: property of what was just said rather than of the clock: a fragment
    #: buys a fragment's silence, a sentence buys a sentence's.
    gap_after_line_s: float = 0.8
    #: The three caps that replace one flat ``min_gap_s``, chosen by the
    #: phase of play the last line was about. Study section 2.3 measures the
    #: gap between utterances three ways over the four aligned matches: 2.8 s
    #: in an attacking move, 4.2 s in build-up, 4.5 s at a dead ball and
    #: 4.6 s at a stoppage. In the box the commentator speaks half again as
    #: fast and says less each time; at a restart the gaps open and the lines
    #: get longer. One rate through both — which is what this had — sounds
    #: too slow in the box and too busy on the halfway line, and Gap 3 of
    #: ``docs/research/real-commentary-corpus.md`` says so in those words.
    #:
    #: ``min_gap_s`` stays as the default for a line filed under no phase at
    #: all, which is the colour seat's and the analyst's.
    min_gap_attacking_s: float = 2.5
    min_gap_build_up_s: float = 4.5
    min_gap_dead_ball_s: float = 5.0
    #: Lines shown back to the model as "the last lines spoken". This is the
    #: whole of what stops it repeating itself; a similarity veto used to sit
    #: behind it and fired zero times in 63 real-clip runs.
    recent_lines: int = 5
    max_words: int = 28
    min_confidence: float = 0.35
    #: Width the caller's frames are sent at. 768 is the point past which a
    #: wide shot costs tokens without adding anything; the question is
    #: whether a shirt number at 1280 is legible where it was four pixels
    #: tall at 768, which is what open-play naming is bounded by.
    frame_width: int = int(os.getenv("CALLER_FRAME_WIDTH", "768"))


@dataclass(frozen=True)
class PhraserConfig:
    """Turning the caller's form into something a commentator would say.

    Every number here used to be a consequence of the measurement in
    ``runs/prompt-name/REAL_COMMENTARY.md``, which was one World Cup final.
    They are now club football's, from
    ``docs/research/real-commentary-corpus.md`` section 1: median eight
    words, 47% of utterances nine words or more, 20% sixteen or more, the
    95th percentile 22 to 27 and the longest in a match 50 and up.
    """

    #: Which model says it, or ``off`` for no phrasing stage at all. Here
    #: rather than read straight from the environment so that a test, a
    #: sweep or a rephrase can turn the stage off without touching the
    #: process it is running in.
    model: str = PHRASER_MODEL
    #: Hard cap on the phrased line, and the caller's cap as well.
    #:
    #: It was 16, chosen as a target rather than a backstop when the
    #: reference was a World Cup final whose 95th percentile is 18 words. The
    #: corpus study measured six club matches: one utterance in five runs to
    #: sixteen words or longer and the 95th percentile is 22 to 27 (study
    #: section 1 and Gap 4), so a cap of 16 removed the top fifth of real
    #: commentary by construction and the v3 traces have no line of nine
    #: words anywhere. 28 is what the caller has always had and it is a
    #: backstop again: the length the phraser should write is in the prompt,
    #: tied to the phase of play, not in this number.
    max_words: int = 28
    #: Real utterances shown per kind in the system prompt. The whole set is
    #: 228 lines and a sample is enough to set a register.
    #:
    #: It was 14 and is 10 because the first two rephrases measured
    #: ``cache_read_input_tokens`` at zero across 33 calls: prompt caching is
    #: asked for and is not firing, so every call pays full price for the
    #: whole prefix and the bill is linear in its length. Ten a kind is about
    #: sixty utterances, which is still a register, and it pays for the rules
    #: that were added after those runs invented a booking and an equaliser.
    #: If the caching is ever fixed this should go back up.
    examples_per_kind: int = 10
    #: Lines shown back as "the last lines spoken", so the voice does not
    #: repeat itself.
    #:
    #: It was 4, on the reasoning that five-word lines make poor context.
    #: The lines are eight words now and the fault the window exists to stop
    #: is still the loudest one in the measurement: 35% of phrased lines open
    #: on a word one of the last five opened on, against 12.5% in the corpus
    #: (``docs/research/real-commentary-corpus.md``, and the register judge
    #: counts it over a window of five). Six is one more than the window the
    #: grader uses, so a repeat the model is shown is a repeat it chose.
    recent_lines: int = 6
    #: Small. The answer is one short sentence and a number.
    max_tokens: int = 256


@dataclass(frozen=True)
class AnalystConfig:
    """The old colour voice: a silence timer that sees frames.

    Superseded by :class:`ColourConfig`, and kept because it still runs when
    ``colour.enabled`` is off. Study section 9, Gap 2 is the case against it:
    a lull is not a phase, and on the Mbappé trace this fired eight seconds
    after the goal call, inside the window the corpus gives to the lead.
    """

    frames: int = 6
    window_s: float = 20.0
    #: Only speaks when nothing has been said for this long.
    lull_s: float = 7.0
    #: Both measured on the first real run, where the analyst spoke five of
    #: seven lines at 40 to 50 words each and editorialised to fill them.
    #: A second voice that talks more than the first is not a second voice.
    min_gap_s: float = 40.0
    max_words: int = 30


@dataclass(frozen=True)
class ColourConfig:
    """The colour seat: when it is offered a turn, and how long a turn is.

    Every number is from ``docs/research/real-commentary-corpus.md``. The
    seat is text-only and event-driven — it never sees a frame and it is
    never on a timer — because section 4.2 counts colour entries per hundred
    utterances by phase and the spread is the whole finding: 2.1 in an
    attacking move against 10.5 at a dead ball, 11.8 at a stoppage and 15.4
    over a replay. Five times less likely while the ball is live.
    """

    #: Which model speaks it, or ``off`` for no colour seat. Read from
    #: ``COLOUR_MODEL`` the way the phraser reads ``PHRASER_MODEL``, so a
    #: test or a sweep can switch seats without touching the environment.
    model: str = COLOUR_MODEL
    #: Whether the runtime prefers this seat over the old silence-timer
    #: analyst. On by default; off restores exactly the runtime that was
    #: there before this file knew about a phase.
    enabled: bool = field(default_factory=lambda: _env_flag("COLOUR_SEAT", True))
    #: Nothing is said inside this long after a goal, shot, save or penalty
    #: form. Section 4.3: the median delay from a big event to the first
    #: colour entry is 21.4 s and only 7% land within six seconds. Those
    #: seconds belong to the lead, who is rebuilding the move and giving the
    #: tally.
    quiet_after_big_s: float = 12.0
    #: The one exception, also 4.3: at a goal 31% of colour entries do arrive
    #: inside six seconds, and what arrives is a reaction fragment — "WELL,
    #: it's the first goal of the game", "Well, well, well." So a goal, and
    #: only a goal, opens a short window.
    goal_reaction_from_s: float = 4.0
    goal_reaction_to_s: float = 8.0
    #: How long after a big event the phase still counts as its aftermath. At
    #: or past this the seat is offered a turn on the clock alone, because by
    #: then the corpus is back to ordinary build-up rates.
    settled_after_big_s: float = 20.0
    #: One turn per this long in build-up. Section 4.2's 5.9 entries per 100
    #: utterances in build-up, against a system that says a line every four
    #: to six seconds, is about one turn a minute; 45 s is that, rounded
    #: towards speaking.
    min_gap_s: float = 45.0
    #: How many caller forms back the phase is read off. Two, because a
    #: single close-up inside a live move is a cutaway and not a stoppage.
    phase_forms: int = 2
    #: A turn is this many utterances. Section 4.4: median run 4, mean 4.4,
    #: 51% run four or more, 20% a single utterance.
    min_utterances: int = 2
    max_utterances: int = 4
    #: Words in one utterance. The corpus's colour entries are 3 to 12 words
    #: each and simply come in sequence; ``AnalystConfig.max_words = 30`` is
    #: one utterance's worth and truncated mid-clause on the real trace.
    max_words: int = 12
    #: Seconds between the utterances of one turn. Section 2.5's median
    #: internal gap at a dead ball is 4.5 s and 4.2 s in build-up, but those
    #: are gaps between *speakers*; inside one held microphone the run is
    #: faster, and the director will cut the tail the moment the lead has
    #: something.
    utterance_gap_s: float = 2.5
    #: No colour utterance sits closer than this to a caller beat — before
    #: it, or after the lead has finished saying it. Two voices on one
    #: channel, and the lead has the ball.
    #:
    #: "After he finishes", not "after the beat is stamped": the first run of
    #: this seat put a colour line 2.0 s after a nine-word lead line, which
    #: at ``WORDS_PER_SECOND`` is 2.8 s before the lead stops talking. The
    #: beat's timestamp is when the line starts.
    clear_of_caller_s: float = 2.0
    #: How long one turn may take from its first utterance to its last.
    #: Pushing utterances clear of the lead stretches a turn, and a thought
    #: that arrives fifteen seconds after the one before it is not the same
    #: turn any more. Anything past this is dropped, which is section 4.6's
    #: hand-back: the colour voice stops mid-thought when the ball moves and
    #: there is no verbal hand-back anywhere in the corpus.
    turn_span_s: float = 10.0
    #: The lead's last lines the seat observes off. Eight rather than the
    #: phraser's four: this seat is looking for what has been true for a
    #: while, not for what it just said.
    lead_lines: int = 8
    #: It never opens cold. Section 4.6: the colour voice comes in after the
    #: lead, and hands back by stopping when the ball moves. Before this many
    #: lead lines exist there is nothing to observe off.
    min_lead_lines: int = 2
    #: Small. The answer is three short sentences and two short lists.
    max_tokens: int = 700


@dataclass(frozen=True)
class PredictorConfig:
    """When to consider speaking at all.

    worldcupvoice speaks on a fixed four-second timer. Silence here is a
    decision: triggers push towards speech, the rate cap pushes back, and
    pressure builds while nothing is said so the system never goes mute.
    """

    tick_s: float = 0.5
    #: Silence longer than this starts pushing the urgency up.
    silence_pressure_after_s: float = 6.0
    silence_forces_at_s: float = 12.0
    #: Mean absolute frame difference above this reads as a camera cut.
    cut_threshold: float = 34.0
    urgency_by_trigger: dict[str, float] = field(
        default_factory=lambda: {
            "board_change": 1.0,
            "camera_cut": 0.35,
            "silence_pressure": 0.3,
            "scheduled": 0.2,
        }
    )


@dataclass(frozen=True)
class GateConfig:
    """The fact gate's strictness."""

    #: A goal is only spoken once the board has changed or a celebration is
    #: visible in the lookahead. Nothing else gets to claim a goal.
    require_board_for_goal: bool = True
    #: Names must match a roster entry at least this well (0-1, token ratio).
    name_match_threshold: float = 0.86
    #: Drop a line rather than trim it if trimming would leave less than this.
    min_words_after_trim: int = 3


@dataclass(frozen=True)
class DirectorConfig:
    """Who speaks, and who gets cut off."""

    #: A beat older than this is stale — the moment has passed, drop it.
    max_beat_age_s: float = 3.5
    #: Goals preempt whatever is being spoken, mid-word.
    preempt_on: tuple[str, ...] = ("goal", "penalty", "card")
    queue_depth: int = 3


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "on", "true", "yes"}


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * t


@dataclass(frozen=True)
class SeatVoice:
    """One seat's voice settings at the two ends of the excitement scale.

    Two points and a straight line between them. Not because the ear is
    linear — it is not — but because a curve with a shape has a shape that
    has to be justified, and nothing has been listened to yet. Two numbers a
    seat is the smallest thing that can be tuned by ear, and the tuning is
    the point.
    """

    #: Falls as the excitement rises. Stability is ElevenLabs' word for how
    #: closely a read hugs the reference: high is even and safe, low lets the
    #: model break pitch and pace, which is what a goal sounds like.
    stability_low: float
    stability_high: float
    #: Rises. Style is how much of the reference voice's own performance is
    #: exaggerated. It is also the setting that costs latency, so the calm
    #: end is kept near zero.
    style_low: float
    style_high: float
    #: Rises, but barely. Past about 1.2 the words start to slur into each
    #: other and it reads as a fast-forward rather than as urgency.
    speed_low: float
    speed_high: float

    def at(self, excitement: float) -> dict[str, float]:
        """The three moving settings at one excitement, clamped to 0..1."""
        t = min(1.0, max(0.0, excitement))
        return {
            "stability": round(_lerp(self.stability_low, self.stability_high, t), 3),
            "style": round(_lerp(self.style_low, self.style_high, t), 3),
            "speed": round(_lerp(self.speed_low, self.speed_high, t), 3),
        }


def _seat(
    prefix: str,
    *,
    stability: tuple[float, float],
    style: tuple[float, float],
    speed: tuple[float, float],
) -> SeatVoice:
    return SeatVoice(
        stability_low=_env_float(f"VOICE_{prefix}_STABILITY_LOW", stability[0]),
        stability_high=_env_float(f"VOICE_{prefix}_STABILITY_HIGH", stability[1]),
        style_low=_env_float(f"VOICE_{prefix}_STYLE_LOW", style[0]),
        style_high=_env_float(f"VOICE_{prefix}_STYLE_HIGH", style[1]),
        speed_low=_env_float(f"VOICE_{prefix}_SPEED_LOW", speed[0]),
        speed_high=_env_float(f"VOICE_{prefix}_SPEED_HIGH", speed[1]),
    )


@dataclass(frozen=True)
class VoiceConfig:
    """How hard a line is said, as a function of how excited it is.

    Every beat has carried an ``excitement`` since the phrasing stage landed
    and nothing read it, so a goal went out at the same library defaults as a
    throw-in. worldcupvoice gets its tone from one fixed set of settings —
    stability 0.35, style 0.35, speed 1.12 — which is a choice about the
    *average* line and is therefore wrong at both ends. These are two sets a
    seat and the line between them.

    The defaults below are a starting guess and are meant to be replaced by
    numbers somebody has listened to; ``scripts/voice_sweep.py`` renders the
    grid to listen to. Every one of them is overridable from the environment,
    and ``VOICE_CURVE=off`` sends no settings at all, which is the voice
    exactly as it was before this existed.

    Read at construction rather than at import, so a test or a sweep can set
    the environment and build a new one.
    """

    #: Play-by-play. Wider than the analyst at both ends: it is the seat that
    #: has to go from naming a throw-in to calling a goal.
    caller: SeatVoice = field(
        default_factory=lambda: _seat(
            "CALLER", stability=(0.55, 0.20), style=(0.15, 0.60), speed=(1.0, 1.15)
        )
    )
    #: Colour. Never shouts: the analyst speaking at a caller's pitch is the
    #: two voices becoming one voice, which is what the second seat exists to
    #: avoid. Excitement here is interest, not volume.
    analyst: SeatVoice = field(
        default_factory=lambda: _seat(
            "ANALYST", stability=(0.60, 0.40), style=(0.10, 0.35), speed=(0.97, 1.05)
        )
    )
    #: How close to the original voice the model stays. Not a function of
    #: excitement: a caller who stops sounding like himself when he shouts is
    #: a different caller, not an excited one.
    similarity_boost: float = field(
        default_factory=lambda: _env_float("VOICE_SIMILARITY_BOOST", 0.8)
    )
    use_speaker_boost: bool = field(default_factory=lambda: _env_flag("VOICE_SPEAKER_BOOST", True))
    #: ``VOICE_CURVE=off`` sends no ``voice_settings`` and each voice plays at
    #: its library defaults. Kept as an escape hatch because a bad curve is
    #: worse than no curve and matchday is not the time to find out.
    on: bool = field(
        default_factory=lambda: (os.getenv("VOICE_CURVE") or "on").strip().lower() != "off"
    )
    #: macOS ``say`` has one dial, words per minute, so the curve collapses to
    #: it. 170 to 210 either side of the 190 it ships with.
    say_rate_low: int = field(default_factory=lambda: _env_int("VOICE_SAY_RATE_LOW", 170))
    say_rate_high: int = field(default_factory=lambda: _env_int("VOICE_SAY_RATE_HIGH", 210))
    #: Punctuation shaping, off by default. See
    #: :mod:`commentary.voice.shaping` for why it is opt-in.
    shaping: bool = field(default_factory=lambda: _env_flag("VOICE_SHAPING", False))

    def seat(self, voice: str) -> SeatVoice:
        return self.analyst if voice == "analyst" else self.caller

    def settings_for(self, voice: str, excitement: float) -> dict[str, float | bool] | None:
        """The ElevenLabs ``voice_settings`` for one beat, or None when off."""
        if not self.on:
            return None
        settings: dict[str, float | bool] = dict(self.seat(voice).at(excitement))
        settings["similarity_boost"] = self.similarity_boost
        settings["use_speaker_boost"] = self.use_speaker_boost
        return settings

    def say_rate(self, excitement: float) -> int:
        """Words per minute for macOS ``say``. Clamped the same way."""
        t = min(1.0, max(0.0, excitement))
        return round(_lerp(self.say_rate_low, self.say_rate_high, t))


@dataclass(frozen=True)
class RestatementConfig:
    """The score-and-clock line a club feed says for viewers joining late.

    ``docs/research/real-commentary-corpus.md`` section 5.3 splits the corpus
    in two. The club-channel feeds restate the score constantly —
    ``bar-mal-2019`` does it on an almost literal five-minute timer, 40 times
    in a match — and the domestic feeds barely do, because the score bug is on
    screen and the viewer can read it. Five minutes is the club number and is
    the default; a broadcast with a permanent bug should turn it off.

    The line is written in code, not by a model, for the reason Gap 8 item 4
    gives: the vocabulary is a small closed set and the handoff's rule is
    already "code writes numbers, the model writes words".
    """

    #: Seconds of *match clock* between restatements. Zero is off.
    every_s: float = float(os.getenv("RESTATEMENT_EVERY_S", "300"))
    #: How close a caller beat may be before the restatement gives way. It is
    #: filler, and filler never speaks over the game.
    clear_of_a_beat_s: float = 3.0
    #: How hard it is said. The flattest thing anybody says in a match.
    excitement: float = 0.1


@dataclass(frozen=True)
class CostConfig:
    """A match that costs more than this stops calling the model."""

    max_usd_per_match: float = float(os.getenv("MAX_USD_PER_MATCH", "35.0"))


@dataclass(frozen=True)
class Settings:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    caller: CallerConfig = field(default_factory=CallerConfig)
    phraser: PhraserConfig = field(default_factory=PhraserConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    colour: ColourConfig = field(default_factory=ColourConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)
    restatement: RestatementConfig = field(default_factory=RestatementConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    cost: CostConfig = field(default_factory=CostConfig)


SETTINGS = Settings()
