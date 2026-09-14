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
    #: build-up in fragments — "De Paul." "Messi, Álvarez." — a median
    #: 2.4s apart, and a two-word line held for four seconds is three
    #: seconds of dead air. Under 1.5s two lines tread on each other.
    min_gap_floor_s: float = 1.5
    #: Breath. Added to however long the last line took, so the gap is a
    #: property of what was just said rather than of the clock: a fragment
    #: buys a fragment's silence, a sentence buys a sentence's.
    gap_after_line_s: float = 0.8
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
class AnalystConfig:
    """The colour voice."""

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


@dataclass(frozen=True)
class CostConfig:
    """A match that costs more than this stops calling the model."""

    max_usd_per_match: float = float(os.getenv("MAX_USD_PER_MATCH", "35.0"))


@dataclass(frozen=True)
class Settings:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    caller: CallerConfig = field(default_factory=CallerConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)
    cost: CostConfig = field(default_factory=CostConfig)


SETTINGS = Settings()
