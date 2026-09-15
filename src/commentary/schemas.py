"""The contracts between the agents. Every hop is a validated object.

These types are the whole interface surface of the system: perception fills
them in, state accumulates them, the fact gate judges them, and the director
schedules them. Nothing crosses a module boundary as a loose dict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

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
    #: The game is stopped and nobody has been penalised: an injury, treatment
    #: on the pitch, a VAR check, the referee holding play up. The card clip
    #: had two lines about a player down being tagged `foul`, because `foul`
    #: was the closest word available, and the grader counted two phantom
    #: fouls for it. Nothing is claimed about anybody, so nothing can be wrong.
    STOPPAGE = "stoppage"
    BUILD_UP = "build_up"
    PASS = "pass"
    CARRY = "carry"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    TACKLE = "tackle"


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


class Sighting(BaseModel):
    """A shirt number or a name the caller could actually read in the picture.

    This is the only record of what the caller read. There used to be a
    second one, ``names_read``, a list of free text, and given two places to
    put a read the caller put everything in that one and left this empty —
    thirteen lines to two on the run that measured it. One field.

    There used to be a ``mark`` here too: a letter tag a local player tracker
    drew over each body, so a read could be tied to a track. The tracker is
    gone — on real clips the caller named more than twice as many players
    without it and bound nearly every sighting, against a third with it —
    and what carries a name from one line to the next is the registry and
    the eight-second carry rule in the runtime, not a body on the picture.

    ``side`` is here because a number on its own names nobody. Both squads
    wear a 5, a 7, a 10 and an 11, and on the penalty clip that cost 18 of 34
    sightings — the caller read the shirt correctly and the bind had no way to
    say which shirt. The side used to be resolved by the kit split, which put
    an Argentina body on France and turned a "26" into Marcus Thuram; the
    model reading the picture does not confuse white stripes with navy, and
    the kit strings it needs are in the team sheets it already has.
    """

    number: int | None = Field(default=None, description="Shirt number, if legible")
    name: str | None = Field(default=None, description="Name on the shirt or a graphic")
    side: Side = Field(
        default=Side.UNKNOWN,
        description=(
            "which team's kit the body wears, from the shirt colours on the team sheet"
        ),
    )


class CallerLine(BaseModel):
    """The caller fills a form, not just a sentence.

    The form feeds match state and the fact gate; ``line`` feeds the voice.
    """

    scene: Scene
    event: Event
    side: Side = Side.UNKNOWN
    team: str | None = Field(default=None, description="Team name, only if legible or inferable")
    sightings: list[Sighting] = Field(
        default_factory=list,
        description="Every shirt number or name you could actually read, one entry per player",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    speak: bool = Field(description="False is a valid answer; silence is allowed")
    line: str = Field(default="", max_length=200)
    #: The one thing in the picture a listener could not have guessed. It is
    #: here because the phrasing stage kept throwing it away: "France drive
    #: into the box, and it's in! Mbappé, off the ground in a flash" came back
    #: as "Mbappé!", which is true, short, and says nothing. Compressing a
    #: line means choosing what survives, and a model that has to choose while
    #: it writes chooses the name every time. So the choice is made here, by
    #: the one agent that actually saw it happen, and handed over as its own
    #: field. Older traces have no such field and the phraser falls back to
    #: reading the detail out of ``line`` itself.
    detail: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "the single most concrete detail of the action: the finish, the direction, "
            "the body part, the distance, the speed; null if none"
        ),
    )


class PhrasedLine(BaseModel):
    """The caller's form, said out loud by somebody who can talk.

    Two fields and no third. Everything factual was settled before this
    stage ran and is checked again after it: the phraser is given a form and
    returns words, and anything it knows that the form did not give it is a
    thing it made up.

    ``line`` may come back empty. The model is told that silence is available
    and that it should reach for a bare surname long before it reaches for
    silence, because by the time the phraser runs the caller and the fact
    gate have both already decided a line is going out. The runtime therefore
    treats an empty line the way it treats an error — it falls back to the
    caller's own words and says so on the bus — rather than losing the line.
    """

    line: str = Field(default="", max_length=200)
    excitement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 for a routine touch, 1 for a goal: the volume this is said at",
    )


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
    #: Where the live edge was when this was produced, on the same clock as
    #: ``video_ts``. Lag is the gap between the two: buffer depth plus however
    #: long the model took. ``created_ts`` is wall time and cannot answer that,
    #: because the two clocks do not share an origin.
    live_ts: float = 0.0
    event: Event = Event.NONE
    urgency: float = 0.0
    #: How hard this should be said, 0 to 1, from the phrasing stage. Zero
    #: when there is no phraser, which is what every speaker still assumes:
    #: the field is carried so that a voice can use it without the runtime
    #: having to change again, and nothing reads it yet.
    excitement: float = 0.0
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


#: What a note is. Three kinds, because the three are used differently: a
#: stat is a count and the gate checks the number, a storyline is a record or
#: a stake, a habit is what this player does every time.
NoteKind = Literal["stat", "storyline", "habit"]


class Note(BaseModel):
    """One short, verifiable thing a commentator can drop into a quiet moment.

    The pack already had ``storylines``, and not one of them has ever reached
    air: they are paragraph-shaped, they are about the fixture rather than
    about a player, and nothing downstream knew which of them belonged to the
    man currently on the ball. A note is the same information cut to the size
    of a clause and filed under a name, so that the runtime can hand the
    phraser the two or three that are about the people on the screen right
    now — and so that the fact gate can check a number said out loud against
    something somebody actually looked up.

    ``about`` is a key, not prose: the exact roster spelling of a player's
    name, or the exact name of one of the two teams. Anything else is a note
    nobody can look up, which is why :func:`~commentary.agents.researcher.check_notes`
    exists and why the researcher's prompt says the word "exact" twice.

    ``text`` is what gets said, or near enough — one clause, under fourteen
    words, no lead-in. "three goals in this tournament", not "Mbappé has
    scored three goals in this tournament, which means...".

    ``source`` is free text and is never spoken. It is there so that a note
    that turns out to be wrong can be traced back to whatever said it was
    right, which is the only defence a pre-match pack has against a confident
    error going to air ninety minutes later.
    """

    about: str = Field(description="A player's exact roster name, or a team's exact name")
    text: str = Field(max_length=120, description="One clause, under 14 words")
    kind: NoteKind = "stat"
    source: str = Field(default="", description="Where this came from; never spoken")

    def __str__(self) -> str:
        return f"{self.about}: {self.text}"


class NoteSheet(BaseModel):
    """Only the notes, for the one-off pass that adds them to a finished pack.

    A pack already on disk has correct squad numbers that cost a research
    call to get right, and asking a model to hand the whole pack back so that
    thirty notes can be added to it is an invitation to lose them. So this is
    the output format of the ``notes`` command: notes and nothing else, merged
    into the pack locally.
    """

    notes: list[Note] = Field(default_factory=list)


class TeamSheet(BaseModel):
    """One team as the researcher found it, before kickoff."""

    name: str
    short: str = ""
    kit: str = Field(default="", description="Shirt colours, so the caller can tell sides apart")
    demonym: str = Field(
        default="",
        description="What this team's players are called collectively: French, Argentine",
    )
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
    #: Short, filed, checkable context: the half of the pack that reaches a
    #: spoken line. Empty by default, so every pack written before notes
    #: existed loads unchanged and simply has nothing to offer.
    notes: list[Note] = Field(
        default_factory=list,
        description="Short verifiable nuggets, each filed under a player or team name",
    )

    @property
    def names(self) -> frozenset[str]:
        """Every name a note may be filed under: both teams and both squads."""
        people = [p.name for sheet in (self.home, self.away) for p in sheet.squad]
        return frozenset([self.home.name, self.away.name, *people])

    def notes_about(self, name: str) -> list[Note]:
        """The notes filed under exactly this name, in pack order."""
        return [note for note in self.notes if note.about == name]

    def team(self, side: Side) -> TeamSheet | None:
        if side is Side.HOME:
            return self.home
        if side is Side.AWAY:
            return self.away
        return None


class WireEvent(BaseModel):
    """One thing a statistician says happened, timed on the match clock.

    A feed knows only match time, so ``video_ts`` starts empty and is filled
    in by :class:`commentary.wire.WireSync` from the board reader's clock
    readings; an event that has not been resolved yet cannot be released.
    The sim's own truth is already on video time and arrives with it set.

    The wire is off by default. It exists as the ceiling row of the ablation
    table — what a play-by-play feed would buy over what the picture gives —
    and never as part of the default runtime.
    """

    event: Event
    side: Side
    player: str | None = None
    recipient: str | None = Field(
        default=None, description="Pass recipient; the fouled player; the player coming off"
    )
    detail: str = Field(default="", description='"yellow", "red", "own goal"')
    home_score: int = 0
    away_score: int = 0
    clock_s: float = Field(description="Match clock, seconds played")
    period: int
    duration_s: float = Field(default=0.0, description="Passes and carries")
    video_ts: float | None = None


class Possession(BaseModel):
    """Who is on the ball, from the wire.

    Separate from :attr:`MatchState.possession`, which is only a side: the
    caller needs a name to say, and a side never gives it one.
    """

    player: str
    side: Side
    since_ts: float
    from_player: str | None = Field(default=None, description="Who passed it, when known")


class NamedEvent(BaseModel):
    """Something the statistician named, close enough to still be worth saying.

    Fouls, offsides, saves, tackles: the things commentary is expected to
    attribute and the picture almost never can. Kept as a short rolling list
    rather than as state, because "who was fouled" stops being news about
    twenty seconds after the whistle.
    """

    event: Event
    side: Side
    player: str | None = None
    recipient: str | None = None
    detail: str = ""
    video_ts: float = 0.0


class Incident(BaseModel):
    """A goal, a card or a substitution, and which source reported it.

    The board and the wire both see goals and they disagree about when: the
    board sees the graphic, the wire saw the ball cross the line. A trace
    that cannot say which one moved the score cannot explain a correction.
    """

    event: Event
    side: Side
    player: str | None
    video_ts: float
    source: str = Field(description='"board" or "wire"')


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
    #: False once the score bug has been gone longer than any replay lasts:
    #: the wrong crop, or a broadcast that carries no bug at all.
    bug_visible: bool = True
    last_events: list[Event] = Field(default_factory=list)
    possession: Side = Side.UNKNOWN
    on_pitch: dict[str, str] = Field(
        default_factory=dict, description="Shirt number to name, as learned from graphics"
    )
    ball: Possession | None = None
    incidents: list[Incident] = Field(default_factory=list)
    named: list[NamedEvent] = Field(default_factory=list)

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
