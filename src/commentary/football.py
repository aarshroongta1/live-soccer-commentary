"""Small football action vocabulary shared by perception and independent labels.

Cues are model-reported evidence, not proof that classification is correct.
Structural validation prevents missing support; held-out labels measure accuracy.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal[
    "pass",
    "carry",
    "dribble",
    "cross",
    "shot",
    "header",
    "save",
    "block",
    "tackle",
    "interception",
    "ball_recovery",
    "clearance",
    "goal",
    "foul",
    "offside",
    "corner",
    "throw_in",
    "free_kick",
    "penalty_awarded",
    "penalty_kick",
    "yellow_card",
    "red_card",
    "substitution",
    "kick_off",
    "goal_kick",
    "unknown",
]
BallPath = Literal["ground", "aerial", "unknown"]
VisualCue = Literal[
    "ball_contact",
    "ball_motion",
    "ball_on_ground",
    "ball_flight",
    "ball_in_net",
    "goal_line_crossing",
    "keeper_contact",
    "player_challenge",
    "restart_setup",
    "throw_in_motion",
    "referee_signal",
    "offside_signal",
    "yellow_card_visible",
    "red_card_visible",
    "substitution_board",
    "on_screen_graphic",
]


class FootballAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ActionType
    certainty: Literal["confirmed", "uncertain"]
    ball_path: BallPath = "unknown"
    cues: list[VisualCue] = Field(default_factory=list, max_length=4)
    support_at_s: list[float] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def supported_claim(self) -> FootballAction:
        if any(not math.isfinite(t) or t < 0 for t in self.support_at_s):
            raise ValueError("support timestamps must be finite and nonnegative")
        if self.support_at_s != sorted(set(self.support_at_s)):
            raise ValueError("support timestamps must be unique and chronological")
        if self.type == "unknown" and self.certainty != "uncertain":
            raise ValueError("unknown actions must remain uncertain")
        if self.certainty == "confirmed" and (not self.cues or not self.support_at_s):
            raise ValueError("confirmed actions require visual cues and supporting frames")
        required: dict[str, set[str]] = {
            "goal": {"ball_in_net", "goal_line_crossing"},
            "foul": {"referee_signal", "on_screen_graphic"},
            "offside": {"offside_signal", "on_screen_graphic"},
            "penalty_awarded": {"referee_signal", "on_screen_graphic"},
            "yellow_card": {"yellow_card_visible", "on_screen_graphic"},
            "red_card": {"red_card_visible", "on_screen_graphic"},
            "substitution": {"substitution_board", "on_screen_graphic"},
        }
        if (
            self.certainty == "confirmed"
            and self.type in required
            and not set(self.cues).intersection(required[self.type])
        ):
            raise ValueError(f"confirmed {self.type} needs direct event-specific evidence")
        if self.ball_path != "unknown":
            if self.type not in {"pass", "cross", "clearance"}:
                raise ValueError(
                    "ball path is currently classified only for passes/crosses/clearances"
                )
            cue = "ball_on_ground" if self.ball_path == "ground" else "ball_flight"
            if cue not in self.cues:
                raise ValueError("ball path requires its corresponding visible cue")
        return self
