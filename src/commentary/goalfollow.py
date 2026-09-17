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
from enum import StrEnum

from commentary.prompts.phraser import GOAL_BEATS, goal_followup_block
from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    KnowledgePack,
    Note,
    Scene,
    Side,
    Sighting,
)
from commentary.state import notes_for
from commentary.threads import Threads


class GoalPhase(StrEnum):
    """The small lifecycle of one verified goal incident."""

    LIVE = "live"
    CELEBRATION = "celebration"
    REPLAY = "replay"
    FINISHED = "finished"


REPLAY_FACT_CATEGORIES = frozenset({"identity", "sequence", "offside", "technique", "finish"})


@dataclass
class GoalIncident:
    """Runtime identity for one goal, separate from the match score.

    ``GoalFollowup`` owns the broadcast beat cadence and ``ReplaySequence``
    owns camera-angle spacing.  This object answers the different question:
    whether an incoming form belongs to the same verified goal at all.
    """

    phase: GoalPhase = GoalPhase.FINISHED
    scorer: str | None = None
    side: Side = Side.UNKNOWN
    goal_ts: float | None = None
    score_spoken: bool = False
    celebration_spoken: bool = False
    replay_facts: set[str] = field(default_factory=set)

    def start(
        self,
        ts: float,
        *,
        scorer: str | None = None,
        side: Side = Side.UNKNOWN,
    ) -> None:
        self.phase = GoalPhase.LIVE
        self.goal_ts = ts
        self.scorer = scorer
        self.side = side
        self.score_spoken = False
        self.celebration_spoken = False
        self.replay_facts.clear()

    @property
    def active(self) -> bool:
        return self.phase is not GoalPhase.FINISHED and self.goal_ts is not None

    def begin_celebration(self) -> None:
        if self.active and self.phase is GoalPhase.LIVE:
            self.phase = GoalPhase.CELEBRATION

    def begin_replay(self) -> None:
        if self.active:
            self.phase = GoalPhase.REPLAY

    def finish(self) -> None:
        self.phase = GoalPhase.FINISHED

    def claim_score(self) -> bool:
        """Claim the one score mention allowed for this incident."""
        if self.score_spoken:
            return False
        self.score_spoken = True
        return True

    def claim_celebration(self) -> bool:
        """Allow at most one celebration line."""
        if self.celebration_spoken:
            return False
        self.celebration_spoken = True
        self.begin_celebration()
        return True

    def category_for(self, line: CallerLine) -> str:
        """Choose one concrete replay fact category from a caller form."""
        text = " ".join((line.line or "", line.detail or "")).casefold()
        if line.event is Event.OFFSIDE or "offside" in text or "onside" in text:
            return "offside"
        actions = list(line.actions)
        if any(beat.action is Action.FINISH for beat in actions) or any(
            (beat.outcome or "").casefold() in {"goal", "scored", "finish"}
            for beat in actions
        ):
            return "finish"
        if any(
            (beat.body_part or beat.delivery or beat.direction or "").strip()
            for beat in actions
        ) or any(word in text for word in ("near post", "first touch", "backheel", "volley")):
            return "technique"
        if len(actions) >= 2 or any(
            beat.action in {Action.PASS, Action.RECEIVE, Action.LAYOFF, Action.CROSS}
            for beat in actions
        ):
            return "sequence"
        if line.sightings or any(
            getattr(beat.actor, "name", None) or getattr(beat.target, "name", None)
            for beat in actions
        ):
            return "identity"
        return "sequence"

    def claim_replay_fact(self, line: CallerLine) -> str | None:
        """Reserve one not-yet-covered replay fact, or return ``None``."""
        category = self.category_for(line)
        if category not in REPLAY_FACT_CATEGORIES or category in self.replay_facts:
            return None
        self.replay_facts.add(category)
        self.begin_replay()
        return category

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
MAX_SYNTH = 4

#: The last beat the model is asked to write. Beat 1 is the call itself.
LAST_BEAT = max(GOAL_BEATS)

#: The beat that is one number about the scorer. Skipped where no clause
#: gives him one: see :meth:`GoalFollowup._skipped`.
SCORER_BEAT = 3

#: The beat that goes back over the move in the past tense, and the one a
#: replay line replaces. ``GOAL_BEATS[4]`` and
#: :func:`~commentary.prompts.phraser.replay_block` ask for the same line —
#: the past-tense rebuild — and the replay is the one with a picture behind
#: it, so a replay inside the window spends this beat and the synthesiser
#: does not write a second rebuild of the same move.
REBUILD_BEAT = 4

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
    #: Has a replay line already rebuilt this move? Beat 4 and a replay line
    #: are the same line — the past-tense account of how the goal was scored
    #: — and the corpus never says it twice. The replay wins because it has
    #: the pictures behind it: the broadcast is showing the move again while
    #: it is described.
    rebuilt_by_replay: bool = False
    _last_said: float = 0.0
    #: Every line that has gone out about this goal, the call first. What the
    #: follow-up beats are told not to say again: on the free-kick trace the
    #: phrase "over the wall, into the top corner" went out on the call at
    #: 21.2 s, on beat 2 at 25.2 and on the rebuild at 32.5, which is one
    #: piece of information said three times in eleven seconds.
    _spoken: list[str] = field(default_factory=list)
    #: The pack this goal was armed with, so that :meth:`beat` can ask
    #: whether beat 3 has anything to say without being handed one.
    _pack: KnowledgePack | None = None
    _moves: list[str] = field(default_factory=list)
    _names: list[str] = field(default_factory=list)
    _sightings: list[Sighting] = field(default_factory=list)
    _actions: list[ActionBeat] = field(default_factory=list)
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

        Two of them can be skipped rather than asked for, and both are the
        same judgement: a beat with nothing to put in it is a beat that writes
        filler. See :meth:`_skipped`.
        """
        if not self.active(ts):
            return None
        return self._next_after(self.beats_said)

    def _next_after(self, said: int) -> int | None:
        """The next beat worth asking for after ``said``, or ``None``."""
        for number in sorted(GOAL_BEATS):
            if number > said and not self._skipped(number):
                return number
        return None

    def _skipped(self, number: int) -> bool:
        """Is this beat one there is nothing to write?

        The rebuild, when a replay line has already been the rebuild: one
        move, one past-tense account of it.

        And the tally, when nothing anybody researched gives the scorer a
        number. Beat 3 is one number about the man who just scored, and asked
        for it with no clause behind it this stage wrote "He knew exactly
        where that was going." four seconds after "He knew it from the moment
        it left his boot." (``runs/rephrased/r4-shape/freekick``). That is not
        a beat, it is the same thought twice, and the corpus's own third beat
        is a figure every time.
        """
        if number == REBUILD_BEAT and self.rebuilt_by_replay:
            return True
        if number != SCORER_BEAT:
            return False
        if self.threads is None and self._pack is None:
            # Nothing to ask. A caller that hands this object no source of
            # notes at all is one that never had beat 3's material anywhere,
            # and it kept the beat before this rule existed: skipping it here
            # would be deciding on no evidence.
            return False
        return not self.tally_notes()

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
        self.saw_actions(line.actions)

    def saw_actions(self, actions: tuple[ActionBeat, ...] | list[ActionBeat]) -> None:
        """Keep structured move facts for the rebuild and synthetic turns."""
        known = {
            (beat.video_ts, beat.action, _identity_name(beat.actor), _identity_name(beat.target))
            for beat in self._actions
        }
        for beat in actions:
            key = (
                beat.video_ts,
                beat.action,
                _identity_name(beat.actor),
                _identity_name(beat.target),
            )
            if key not in known:
                self._actions.append(beat.model_copy(deep=True))
                known.add(key)
            for identity in (beat.actor, beat.target):
                name = _identity_name(identity)
                if name and name not in self._names:
                    self._names.append(name)
            description = _describe_action(beat)
            if description and description not in self._moves:
                self._moves.append(description)
        self._actions.sort(key=lambda beat: beat.video_ts if beat.video_ts is not None else 0.0)

    def arm(
        self, ts: float, line: CallerLine, spoken: str, pack: KnowledgePack | None = None
    ) -> None:
        """A goal line has passed the gate. Start the window."""
        self.armed_at = ts
        self.side = line.side
        self.beats_said = 1
        self.synthesised = 0
        self.rebuilt_by_replay = False
        self._spoken = [spoken.strip()] if spoken.strip() else []
        self._pack = pack
        self._last_said = ts
        self._moves = []
        self._names = []
        self._sightings = list(line.sightings)
        self._actions = []
        self._scene = line.scene
        self.scorer = _scorer(line, spoken, pack)
        self.saw_form(line)

    def said(self, ts: float, text: str = "") -> None:
        """A follow-up line went out. Move to the next beat, and keep the words.

        ``text`` is what actually reached air, and it is kept for the same
        reason the call's own words are: the next beat is shown all of it and
        told not to say any of it again.

        The counter moves to the beat that was actually due rather than by
        one, because a skipped beat is spent without being said: without
        that, a window with no tally clause in it would offer beat 3 again on
        every line for thirty seconds.
        """
        if not self.active(ts):
            return
        due = self._next_after(self.beats_said)
        self.beats_said = due if due is not None else self.beats_said + 1
        self._last_said = ts
        self.remember(text)

    def remember(self, text: str) -> None:
        """Keep a line that has gone out about this goal, without spending a beat."""
        said = text.strip()
        if said and said not in self._spoken:
            self._spoken.append(said)

    @property
    def spoken(self) -> list[str]:
        """The call, then every line said about this goal since, in order."""
        return list(self._spoken)

    @property
    def call(self) -> str:
        """The words the goal was called with, or ``""`` before one is armed."""
        return self._spoken[0] if self._spoken else ""

    def rebuilt(self, ts: float, text: str = "") -> None:
        """A replay line has gone out over this goal. Beat 4 is spent.

        Not :meth:`said`, which would spend whichever beat happened to be
        next. A replay line arriving four seconds after the call is the
        past-tense rebuild whatever the counter says, and the celebration and
        the tally are still owed — so the flag is set, the clock for the next
        beat is pushed back, and the count is left where it was.
        """
        if not self.active(ts):
            return
        self.rebuilt_by_replay = True
        self._last_said = ts
        self.remember(text)

    def close(self) -> None:
        """Forget the goal. Called when a line that is not about it goes out."""
        self.armed_at = None

    # -- what it produces -------------------------------------------------

    def block(self, ts: float, pack: KnowledgePack | None = None) -> str:
        """The instruction block for the phraser, or ``""`` outside the window."""
        beat = self.beat(ts)
        if beat is None or self.armed_at is None:
            return ""
        # Beat 3 sees the tally and nothing else, where there is one. The
        # fallback is for the one case :meth:`_skipped` cannot judge — a
        # window armed with no pack behind it, where showing the man's other
        # clauses beats showing him none.
        notes = self.scorer_notes(ts, pack)
        if beat == SCORER_BEAT:
            notes = self.tally_notes(ts, pack) or notes
        return goal_followup_block(
            beat,
            since_s=ts - self.armed_at,
            scorer=self.scorer,
            notes=notes,
            moves=self._moves,
            names=self._names,
            said=self._spoken,
        )

    def tally_notes(self, ts: float | None = None, pack: KnowledgePack | None = None) -> list[Note]:
        """The clauses beat 3 may use: a number about the scorer, and nothing else.

        A note the researcher marked as counting something, and nothing else,
        because :class:`commentary.tallies.Tallies` has already moved its
        figure for the goal that has just gone in — "five goals in this
        tournament" is seven by the time the seventh is scored, and no other
        clause in the pack is true of the match as it stands.

        An ordinal inside a storyline is not a tally, which is the whole of
        the fault: offered "chasing a second World Cup" the line came back
        "That's his second World Cup goal, and he's chasing a second title",
        a count nobody ever made out of a number that was about something
        else. ``clips/pack-argfra-2022-researched.json`` marks six notes this
        way and ``clips/pack-7576.json`` marks none, which is why Ronaldo's
        goal has no third beat and Mbappé's has one.
        """
        notes = self.scorer_notes(
            self._last_said if ts is None else ts, self._pack if pack is None else pack
        )
        return [note for note in notes if note.counts]

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
        due = self.beats_said
        while len(times) < room and at <= ends - BEAT_GAP_S:
            nxt = self._next_after(due)
            if nxt is None:
                break
            due = nxt
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
            actions=[beat.model_copy(deep=True) for beat in self._actions],
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
    finishers = [
        _identity_name(beat.actor)
        for beat in line.actions
        if beat.action is Action.FINISH and _identity_name(beat.actor)
    ]
    for name in finishers:
        if name and name.rsplit(" ", 1)[-1].lower() in said:
            return name
    if finishers:
        return finishers[0]
    names = [(s.name or "").strip() for s in line.sightings]
    for name in names:
        if name and name.rsplit(" ", 1)[-1].lower() in said:
            return name
    named = next((name for name in names if name), None)
    if named:
        return named
    return _from_the_roster(line, said, pack)


def _identity_name(identity: object) -> str:
    name = getattr(identity, "name", None)
    return name.strip() if isinstance(name, str) else ""


def _describe_action(beat: ActionBeat) -> str:
    actor = _identity_name(beat.actor)
    target = _identity_name(beat.target)
    parts = [actor, beat.action.value]
    if target:
        parts.append(f"to {target}")
    if beat.delivery:
        parts.append(beat.delivery)
    if beat.destination_zone:
        parts.append(f"to the {beat.destination_zone}")
    if beat.outcome:
        parts.append(beat.outcome)
    return " ".join(part for part in parts if part).strip()


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
