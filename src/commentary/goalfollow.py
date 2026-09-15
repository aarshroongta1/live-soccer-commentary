"""The thirty seconds after a goal, which is where this system went quiet.

``docs/research/real-commentary-corpus.md`` section 2.4 measured 19 goals
across four aligned matches. In the 30 s after the goal event, the median is
**7 utterances and 60 words**, the gap from the call to the next line is 2.0 s
and the longest gap anywhere in the window is 7.1 s. That is the fastest
sustained talking in a match. The five seconds of silence the earlier study
saw after the Di María goal is a World Cup final's pause for a crowd, and no
league game does it.

What this system did instead, on the Mbappé trace: two lines each goal, seven
and nine words, the second one refused by the gate both times, and nothing
after. Section 8.4 lists the beats the corpus fills and this fills two of
them:

1. a fragment of anticipation before the ball crosses    — the caller has this
2. the strike, one to three words                        — the caller has this
3. repetition of the name, often with an intensifier     — **missing**
4. the score, its own short utterance, within 5 s        — code writes it now
5. a number about the scorer, within about 15 s          — **missing**
6. the move rebuilt in past tense, two or three names    — **missing**
7. the colour voice, 4 to 26 seconds in                  — the colour seat has this

:class:`GoalFollowup` is the state that supplies 3, 5 and 6. It arms when a
goal line passes the gate, lasts thirty seconds, and hands the phraser a
different instruction block on every line inside the window saying which beat
is due. It also answers the two questions the *scoreline* rule needs: whether
this line is the goal call (and therefore the one line that gets a number
appended) or a line after it (and therefore one that gets any number stripped
and none added).

Where the beats come from. Mostly from the caller's own subsequent calls,
which on a real goal are plentiful: the celebration, the ball carried back,
the replay. Where the caller has gone quiet — and on the Mbappé trace it goes
quiet for 24 s after the first goal, because every call in between was a
replay it chose not to speak over — :meth:`synth_times` says where to put up
to two extra phraser calls, built from the descriptions the caller wrote and
did not say. Nothing in a synthesised call is new perception: the form carries
the goal line's own sightings and the caller's own words about the move.

Both the runtime and the offline rephrase drive this same object, so a beat
that exists in one exists in the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from commentary.prompts.phraser import GOAL_BEATS, goal_followup_block
from commentary.schemas import CallerLine, Event, KnowledgePack, Note, Scene, Side, Sighting
from commentary.state import notes_for
from commentary.threads import Threads

#: How long the window lasts. Section 2.4 measures the 30 s after the goal,
#: and the seven utterances are all inside it.
FOLLOWUP_S = 30.0

#: The shortest gap between two follow-up lines. The corpus's own gap from the
#: call to the next line is a median 2.0 s.
BEAT_GAP_S = 2.0

#: How long a gap in the caller's own calls has to be before this fills it.
#: The corpus's longest gap anywhere inside the window is 7.1 s, so six is
#: where a silence stops looking like commentary.
SYNTH_AFTER_S = 6.0

#: Where a synthesised call goes: four seconds after the last line, which is
#: inside the corpus's longest internal gap with room to spare.
SYNTH_GAP_S = 4.0

#: At most this many, per goal. Two extra calls is at most $0.002 of Haiku and
#: it is the difference between a two-line goal and a five-line one.
MAX_SYNTH = 2

#: The last beat the model is asked to write. Beat 1 is the call itself.
LAST_BEAT = max(GOAL_BEATS)

#: How long after the window closes a scoreline-and-clock restatement still
#: stays out of the way. The corpus's celebration does not stop dead at the
#: window's edge, and a restatement landing right on it reads as stepping on
#: the goal — the runtime and the offline rephrase both drive this, and a
#: restatement is the one other thing in the system that writes to the same
#: thirty seconds.
RESTATEMENT_GRACE_S = 30.0


def blocks_restatement(ts: float, armed_at: float, window_s: float = FOLLOWUP_S) -> bool:
    """Is ``ts`` inside a goal's window, or the grace period right after it?

    A free function as well as :meth:`GoalFollowup.blocks_restatement`
    because the offline rephrase has to check every goal a whole trace has
    armed by the time it runs the restatement pass, not only the one the
    live instance still remembers — a second goal replaces the first in
    :attr:`GoalFollowup.armed_at`, and a match with two goals has moved on
    from the first by the time this runs.
    """
    return 0.0 <= ts - armed_at <= window_s + RESTATEMENT_GRACE_S


@dataclass
class GoalFollowup:
    """What has been said about the goal being celebrated, and what is due.

    One instance per runtime or rephrase, re-armed on each goal rather than
    one per goal: a second goal thirty seconds after the first replaces the
    first, which is what a commentator does too.
    """

    window_s: float = FOLLOWUP_S
    #: The pack's notes with a memory and a running count behind them, when
    #: the runtime or the rephrase has one. Beat 3 asks it for the scorer's
    #: clauses rather than looking them up itself, so that a number the goal
    #: has just changed is the number that goes out, and so that saying it
    #: here counts against the thread the same way as saying it anywhere else.
    threads: Threads | None = None
    #: When the goal-calling line went out, or ``None`` if no goal is live.
    armed_at: float | None = None
    #: The side that scored, as the caller filed it. Only used for reporting;
    #: the scoreline arithmetic takes the side from the form it is judging.
    side: Side = Side.UNKNOWN
    #: Who scored, as far as anything can tell: the first name on the goal
    #: form that the spoken line actually used, else the first name read.
    scorer: str | None = None
    beats_said: int = 0
    synthesised: int = 0
    _last_said: float = 0.0
    _moves: list[str] = field(default_factory=list)
    _names: list[str] = field(default_factory=list)
    _sightings: list[Sighting] = field(default_factory=list)
    _scene: Scene = Scene.LIVE_PLAY

    # -- what the window knows -------------------------------------------

    def active(self, ts: float) -> bool:
        """Is a goal still being celebrated at this moment?"""
        if self.armed_at is None:
            return False
        return 0.0 <= ts - self.armed_at <= self.window_s

    def blocks_restatement(self, ts: float) -> bool:
        """Is a scoreline-and-clock restatement too close to a goal at ``ts``?

        True through the window and for :data:`RESTATEMENT_GRACE_S` after it
        closes — wider than :meth:`active`, which the phraser's own beats use
        and which stops the instant the window does.
        """
        return self.armed_at is not None and blocks_restatement(ts, self.armed_at, self.window_s)

    def beat(self, ts: float) -> int | None:
        """Which beat is due, or ``None`` if the window is closed or spent.

        Beat 1 is the call and is never asked for here: the window only opens
        once it has gone out. Beats run to :data:`LAST_BEAT` and then stop,
        because four short lines plus the call plus the colour seat's two is
        already the corpus's seven.
        """
        if not self.active(ts):
            return None
        due = self.beats_said + 1
        return due if due in GOAL_BEATS else None

    def due(self, ts: float) -> bool:
        """Has enough quiet passed since the last line for the next beat?"""
        return self.beat(ts) is not None and ts - self._last_said >= BEAT_GAP_S

    def is_the_call(self, ts: float) -> bool:
        """Is a goal claimed at this moment the *first* line of that goal?

        The one question the scoreline rule turns on. True outside an armed
        window, which is a new goal being called; false inside one, which is
        the celebration of a goal whose number has already gone out.
        """
        return not self.active(ts)

    # -- what it is told --------------------------------------------------

    def saw_form(self, line: CallerLine) -> None:
        """Keep the caller's description of the move, spoken or not.

        Deliberately fed every form, including the ones the caller chose not
        to speak. After the Mbappé penalty the caller filled in four forms in
        a row and said none of them — they were replays — and their text is
        the only account anywhere of how the goal was scored. Beat 4 is a
        past-tense rebuild of the move, and this is the move.
        """
        text = (line.line or "").strip()
        if text and text not in self._moves:
            self._moves.append(text)
        detail = (line.detail or "").strip()
        if detail and detail not in self._moves:
            self._moves.append(detail)
        for sighting in line.sightings:
            name = (sighting.name or "").strip()
            if name and name not in self._names:
                self._names.append(name)

    def arm(
        self, ts: float, line: CallerLine, spoken: str, pack: KnowledgePack | None = None
    ) -> None:
        """A goal line has passed the gate. Start the window."""
        self.armed_at = ts
        self.side = line.side
        self.beats_said = 1
        self.synthesised = 0
        self._last_said = ts
        self._moves = []
        self._names = []
        self._sightings = list(line.sightings)
        self._scene = line.scene
        self.scorer = _scorer(line, spoken, pack)
        self.saw_form(line)

    def said(self, ts: float) -> None:
        """A follow-up line went out. Move to the next beat."""
        if not self.active(ts):
            return
        self.beats_said += 1
        self._last_said = ts

    def close(self) -> None:
        """Forget the goal. Called when a line that is not about it goes out."""
        self.armed_at = None

    # -- what it produces -------------------------------------------------

    def block(self, ts: float, pack: KnowledgePack | None = None) -> str:
        """The instruction block for the phraser, or ``""`` outside the window."""
        beat = self.beat(ts)
        if beat is None or self.armed_at is None:
            return ""
        notes = self.scorer_notes(ts, pack)
        return goal_followup_block(
            beat,
            since_s=ts - self.armed_at,
            scorer=self.scorer,
            notes=notes,
            moves=self._moves,
            names=self._names,
        )

    def scorer_notes(self, ts: float, pack: KnowledgePack | None = None) -> list[Note]:
        """The researched clauses about the man who has just scored.

        One selection function with the quiet moments, not two. Where a
        :class:`~commentary.threads.Threads` exists it decides, with the goal
        payoff on: his running count first, already moved by the goal that has
        just gone in, and a clause said a minute ago allowed back because the
        number in it is not the number any more. Without one — an older caller
        of this, or a run with no pack — it falls back to the plain lookup by
        name, which is what this did before threads existed.
        """
        if not self.scorer:
            return []
        if self.threads is not None:
            return [item.note for item in self.threads.offer([self.scorer], ts=ts, payoff=True)]
        return notes_for(pack, [self.scorer])

    def synth_times(self, after: float, until: float | None) -> list[float]:
        """Where to put extra phraser calls because the caller has gone quiet.

        ``after`` is the line just spoken and ``until`` is the caller's next
        one, or ``None`` if there is not another inside the window. A gap
        shorter than :data:`SYNTH_AFTER_S` is left alone: the caller is about
        to fill it itself, and two voices on the same moment is worse than
        one.
        """
        if self.armed_at is None:
            return []
        ends = min(self.armed_at + self.window_s, until if until is not None else 1e9)
        if ends - after < SYNTH_AFTER_S:
            return []
        times: list[float] = []
        at = after + SYNTH_GAP_S
        room = MAX_SYNTH - self.synthesised
        while len(times) < room and at <= ends - BEAT_GAP_S:
            if self.beats_said + 1 + len(times) not in GOAL_BEATS:
                break
            times.append(round(at, 3))
            at += SYNTH_GAP_S
        return times

    def synthetic(self) -> CallerLine:
        """A form for a call the caller never made, carrying nothing new.

        The event is ``goal`` because the moment is a goal, the sightings are
        the goal line's own — already checked by the gate once — and the text
        is the caller's description of the move. The phraser is writing about
        something that has been seen and judged; it is not being handed a
        picture.
        """
        return CallerLine(
            scene=self._scene,
            event=Event.GOAL,
            side=self.side,
            sightings=list(self._sightings),
            confidence=1.0,
            speak=True,
            line=(self._moves[0] if self._moves else "")[:200],
            detail=None,
        )


def _scorer(line: CallerLine, spoken: str, pack: KnowledgePack | None = None) -> str | None:
    """Whose goal it is, as far as the form and the words agree.

    The name the spoken line actually used wins, because that is the man the
    listener has just heard about. Failing that, the first name the caller
    could read, which on a goal is almost always the scorer: the broadcast
    cuts to him.

    And failing *that* — which is not hypothetical, because on the Mbappé
    penalty every sighting on the goal form came back with a shirt number and
    no name at all, so the goal that was called "Mbappé steps up and strikes
    it" was credited to nobody — the roster. The line names him even when the
    picture could not: the first player of the *scoring side* whose name
    appears in the words that went out. Restricting it to that side is what
    makes it safe, because the other name in a goal call is the goalkeeper's,
    and "buried past Martínez" would otherwise credit the man who was beaten.
    """
    said = spoken.lower()
    names = [(s.name or "").strip() for s in line.sightings]
    for name in names:
        if name and name.rsplit(" ", 1)[-1].lower() in said:
            return name
    named = next((name for name in names if name), None)
    if named:
        return named
    return _from_the_roster(line, said, pack)


def _from_the_roster(line: CallerLine, said: str, pack: KnowledgePack | None) -> str | None:
    """The first man of the scoring side the spoken line names, if any."""
    team = pack.team(line.side) if pack is not None else None
    if team is None:
        return None
    found: list[tuple[int, str]] = []
    for player in team.squad:
        surname = player.surname.lower()
        where = said.find(surname)
        if surname and where >= 0:
            found.append((where, player.name))
    return min(found)[1] if found else None
