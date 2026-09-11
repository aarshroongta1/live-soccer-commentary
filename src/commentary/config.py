"""Runtime settings. Everything tunable in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Models (see docs: claude-api model ids).
CALLER_MODEL = "claude-sonnet-5"
BOARD_MODEL = "claude-haiku-4-5"
ANALYST_MODEL = "claude-opus-5"
RESEARCHER_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class CaptureConfig:
    """Screen capture and the delay buffer that sits behind it."""

    device: str = os.getenv("AVFOUNDATION_DEVICE", "1:0")
    width: int = 1280
    height: int = 720
    fps: int = 15
    #: How far the narration cursor trails the live edge. The headline
    #: experiment sweeps this at 0, 2, 4, 8 seconds.
    delay_s: float = 8.0
    #: Seconds of frames kept behind the cursor, for lookback in prompts.
    history_s: float = 6.0

    @property
    def buffer_frames(self) -> int:
        return int((self.delay_s + self.history_s) * self.fps) + 1


CAPTURE = CaptureConfig()
