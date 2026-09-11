"""The contracts between the agents. Every hop is a validated object."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Scene(StrEnum):
    """What the caller thinks it is looking at."""

    LIVE_PLAY = "live_play"
    REPLAY = "replay"
    CLOSE_UP = "close_up"
    CROWD = "crowd"
    STOPPAGE = "stoppage"
    GRAPHIC = "graphic"


class Event(StrEnum):
    """Events readable from the picture alone."""

    NONE = "none"
    GOAL = "goal"
    SHOT = "shot"
    SAVE = "save"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    PENALTY = "penalty"
    FOUL = "foul"
    OFFSIDE = "offside"
    THROW_IN = "throw_in"
    CARD = "card"
    SUBSTITUTION = "substitution"
    KICKOFF = "kickoff"
    BUILD_UP = "build_up"


class BoardRead(BaseModel):
    """One look at the score bug. Absent bug means we are in a replay."""

    bug_visible: bool
    home_score: int | None = None
    away_score: int | None = None
    clock: str | None = Field(default=None, description="As shown, e.g. '37:12' or '45+2'")
    confidence: float = Field(ge=0.0, le=1.0)


class CallerLine(BaseModel):
    """The caller fills a form, not just a sentence.

    The form feeds match state and the fact gate; ``line`` feeds the voice.
    """

    scene: Scene
    event: Event
    team: str | None = Field(default=None, description="Team name, only if legible or inferable")
    names_read: list[str] = Field(
        default_factory=list,
        description="Names or shirt numbers actually visible in the frames or a graphic",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    speak: bool = Field(description="False is a valid answer; silence is allowed")
    line: str = Field(default="", max_length=200)


class MatchState(BaseModel):
    """Everything the system believes, built only from the board and the caller."""

    home: str
    away: str
    home_score: int = 0
    away_score: int = 0
    clock: str | None = None
    period: int = 1
    in_replay: bool = False
    last_events: list[Event] = Field(default_factory=list)


class GateVerdict(BaseModel):
    """Why a line was allowed through, or was not."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)
