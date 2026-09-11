"""The contracts between the agents. Every hop is a validated object.

These types are the whole interface surface of the system: perception fills
them in, state accumulates them, the fact gate judges them, and the director
schedules them. Nothing crosses a module boundary as a loose dict.
"""

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


#: Events worth interrupting anything else for.
BIG_EVENTS = frozenset({Event.GOAL, Event.PENALTY, Event.CARD, Event.SAVE})


class Side(StrEnum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


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
    side: Side = Side.UNKNOWN
    team: str | None = Field(default=None, description="Team name, only if legible or inferable")
    names_read: list[str] = Field(
        default_factory=list,
        description="Names or shirt numbers actually visible in the frames or a graphic",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    speak: bool = Field(description="False is a valid answer; silence is allowed")
    line: str = Field(default="", max_length=200)


class Angle(StrEnum):
    """What the analyst is about to talk about, so the director can vary it."""

    TACTICS = "tactics"
    FORM = "form"
    PLAYER = "player"
    STAKES = "stakes"
    HISTORY = "history"
    MOMENTUM = "momentum"


class AnalystLine(BaseModel):
    """The colour voice. Wider window, slower rate, cites what it leans on."""

    angle: Angle
    cites: list[str] = Field(
        default_factory=list,
        description="Facts from the knowledge pack or match state this leans on",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    speak: bool
    line: str = Field(default="", max_length=280)


class Trigger(StrEnum):
    """Why the system considered speaking at this instant."""

    WHISTLE = "whistle"
    ROAR = "roar"
    CAMERA_CUT = "camera_cut"
    BOARD_CHANGE = "board_change"
    SILENCE_PRESSURE = "silence_pressure"
    SCHEDULED = "scheduled"


class SpeakDecision(BaseModel):
    """The speak predictor's verdict for one tick."""

    should_call: bool
    triggers: list[Trigger] = Field(default_factory=list)
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class Voice(StrEnum):
    CALLER = "caller"
    ANALYST = "analyst"


class Beat(BaseModel):
    """One thing to say, on its way to a voice.

    ``video_ts`` is the buffer time the line describes, not the time it was
    produced — the eval aligns on it, and the UI shows it next to the frame.
    """

    id: str
    voice: Voice
    text: str
    video_ts: float
    created_ts: float
    event: Event = Event.NONE
    urgency: float = 0.0
    triggers: list[Trigger] = Field(default_factory=list)
    preemptable: bool = True


class GateVerdict(BaseModel):
    """Why a line was allowed through, or was not."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    line: str = Field(default="", description="The line as it should be spoken, possibly trimmed")


class Player(BaseModel):
    name: str
    number: int | None = None
    position: str | None = None

    @property
    def surname(self) -> str:
        return self.name.rsplit(" ", 1)[-1]


class TeamSheet(BaseModel):
    """One team as the researcher found it, before kickoff."""

    name: str
    short: str = ""
    kit: str = Field(default="", description="Shirt colours, so the caller can tell sides apart")
    formation: str | None = None
    manager: str | None = None
    starters: list[Player] = Field(default_factory=list)
    bench: list[Player] = Field(default_factory=list)

    @property
    def squad(self) -> list[Player]:
        return [*self.starters, *self.bench]


class KnowledgePack(BaseModel):
    """Everything prepared before kickoff. The only outside information allowed.

    Assembled once by the researcher and frozen at kickoff; the caller and the
    analyst may cite it, and the fact gate checks names against it.
    """

    home: TeamSheet
    away: TeamSheet
    competition: str = ""
    venue: str = ""
    kickoff: str = ""
    storylines: list[str] = Field(default_factory=list)
    form: dict[str, str] = Field(default_factory=dict)
    key_matchups: list[str] = Field(default_factory=list)

    def team(self, side: Side) -> TeamSheet | None:
        if side is Side.HOME:
            return self.home
        if side is Side.AWAY:
            return self.away
        return None


class MatchState(BaseModel):
    """Everything the system believes, built only from the board and the caller."""

    home: str
    away: str
    home_score: int = 0
    away_score: int = 0
    clock: str | None = None
    clock_s: float | None = Field(default=None, description="Clock parsed to seconds played")
    period: int = 1
    in_replay: bool = False
    last_events: list[Event] = Field(default_factory=list)
    possession: Side = Side.UNKNOWN
    on_pitch: dict[str, str] = Field(
        default_factory=dict, description="Shirt number to name, as learned from graphics"
    )

    @property
    def scoreline(self) -> str:
        return f"{self.home} {self.home_score}-{self.away_score} {self.away}"


class GroundTruthEvent(BaseModel):
    """One thing that actually happened, for grading only. Never seen at runtime."""

    video_ts: float
    event: Event
    side: Side = Side.UNKNOWN
    player: str | None = None
    home_score: int = 0
    away_score: int = 0
