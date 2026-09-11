"""A match that never happened, played out the same way every time.

There is no broadcast clip to develop against yet, and waiting for one blocks
the board reader, the caller, the fact gate and the eval all at once. So we
invent a match: a seeded script of events, the position of the ball and of
twenty-two players at any instant, and the record of what actually happened.
That record is the thing every other stage gets graded against, and it is the
one piece of the sim no agent is ever allowed to see.

The script is built once at construction and then read by random access.
Nothing here steps forward in time, because the renderer, the audio and the
eval all want to ask about an arbitrary timestamp without replaying the match
to get there.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import dataclass

from commentary.schemas import (
    Event,
    GroundTruthEvent,
    KnowledgePack,
    Player,
    Scene,
    Side,
    TeamSheet,
)

#: Pitch in metres. The renderer works in these units, so a shot that looks
#: like it came from the edge of the box came from the edge of the box.
PITCH_L = 105.0
PITCH_W = 68.0

#: 4-4-2 as (fraction of the pitch from your own goal line, fraction across).
FORMATION: tuple[tuple[float, float], ...] = (
    (0.04, 0.50),
    (0.20, 0.16),
    (0.17, 0.38),
    (0.17, 0.62),
    (0.20, 0.84),
    (0.44, 0.14),
    (0.40, 0.40),
    (0.40, 0.60),
    (0.44, 0.86),
    (0.66, 0.38),
    (0.66, 0.62),
)

POSITIONS: tuple[str, ...] = (
    "GK",
    "RB",
    "CB",
    "CB",
    "LB",
    "RM",
    "CM",
    "CM",
    "LM",
    "ST",
    "ST",
)
BENCH_POSITIONS: tuple[str, ...] = ("GK", "CB", "RB", "CM", "CM", "LW", "ST")

_FIRST_NAMES: tuple[str, ...] = (
    "Owen",
    "Mateo",
    "Kasper",
    "Dimitri",
    "Rafa",
    "Nils",
    "Tobias",
    "Jonas",
    "Elias",
    "Marek",
    "Ivan",
    "Luka",
    "Andres",
    "Bruno",
    "Milos",
    "Teo",
    "Felix",
    "Samir",
    "Yannick",
    "Dario",
    "Emre",
    "Noah",
    "Gustav",
    "Renzo",
    "Aleks",
    "Pierre",
    "Joel",
    "Sandro",
    "Kofi",
    "Hugo",
    "Lars",
    "Vito",
    "Adnan",
    "Ciaran",
    "Diego",
    "Mio",
)

_SURNAMES: tuple[str, ...] = (
    "Vance",
    "Okonjo",
    "Lindqvist",
    "Moreau",
    "Barros",
    "Halvorsen",
    "Petrov",
    "Duarte",
    "Kestrel",
    "Novak",
    "Ferreira",
    "Adeyemi",
    "Kowalczyk",
    "Brandt",
    "Salvatore",
    "Mwangi",
    "Ibarra",
    "Lund",
    "Castellan",
    "Haddad",
    "Rinaldi",
    "Sorensen",
    "Ashgrove",
    "Delacroix",
    "Varga",
    "Okafor",
    "Bellini",
    "Cruz",
    "Thorne",
    "Marchetti",
    "Ilves",
    "Dembele",
    "Rask",
    "Quintero",
    "Nakamura",
    "Voss",
    "Fontaine",
    "Olabisi",
    "Strand",
    "Perretti",
    "Grimaldi",
    "Hale",
    "Bergman",
    "Tanaka",
    "Silvestre",
    "Amory",
    "Kalu",
    "Weller",
    "Rossi",
    "Andric",
    "Fenwick",
    "Osei",
    "Kraven",
    "Lombardi",
    "Yilmaz",
    "Drakos",
    "Cassidy",
    "Monteiro",
    "Bjorn",
    "Escobar",
    "Pavlenko",
    "Renard",
    "Ito",
    "Sandell",
    "Alvarez",
    "Holloway",
    "Nakhla",
    "Sturm",
    "Oyelaran",
    "Vidal",
    "Mendoza",
    "Klose",
)

#: (name, three-letter code, kit description). The renderer reads the first
#: colour word out of the kit string, so the dots match what the notes claim.
HOME_IDENTITY = ("Ashcombe Rangers", "ASH", "red shirts, white shorts")
AWAY_IDENTITY = ("Verity Athletic", "VER", "blue shirts, blue shorts")


@dataclass(frozen=True)
class Dot:
    """One player as the camera sees them: a shirt colour and a number."""

    side: Side
    number: int
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class SimState:
    """Everything true about the broadcast at one video timestamp.

    This is the sim's ground truth for a single instant, which makes it both
    the renderer's input and the oracle's answer key. Two things are kept
    apart on purpose: ``scene`` is what the director of the broadcast chose to
    show, and ``event`` is what is happening in the match. A replay of a goal
    is ``REPLAY`` plus ``GOAL``, and a system that conflates the two will
    announce a second goal that never happened.
    """

    ts: float
    scene: Scene
    event: Event
    period: int
    clock: str
    clock_s: float
    home_score: int
    away_score: int
    possession: Side
    ball: tuple[float, float]
    players: tuple[Dot, ...]
    focus: tuple[float, float]
    zoom: float
    graphic: Player | None = None
    graphic_side: Side = Side.UNKNOWN

    @property
    def in_replay(self) -> bool:
        return self.scene is Scene.REPLAY

    def dot(self, side: Side, number: int) -> Dot | None:
        for d in self.players:
            if d.side is side and d.number == number:
                return d
        return None


#: Events that settle a move one way or the other. Everything else in the
#: script is a restart, and nobody is holding their breath over a throw-in.
DECIDING = frozenset({Event.GOAL, Event.SAVE, Event.PENALTY})

#: How far ahead of a decisive moment play counts as still in the balance,
#: and how long after it the picture makes the answer plain. Both are
#: modelling assumptions; see :meth:`MatchSim.outcome_at`.
PENDING_HORIZON_S = 5.0
OUTCOME_VISIBLE_AFTER_S = 1.0


@dataclass(frozen=True)
class Outcome:
    """How a move in the balance ends, and whether it has ended yet.

    ``pending`` is the whole point. Most of a match is not in the balance, and
    a commentator describing a sideways pass in midfield is not guessing at
    anything. It is the few seconds after a shot leaves a boot that separate a
    caller with lookahead from one without, and those are the only seconds
    where the delay can earn its keep.
    """

    event: Event
    ts: float
    pending: bool

    def known_by(self, live_ts: float) -> bool:
        """Whether a caller seeing up to ``live_ts`` can see how this ends."""
        return not self.pending or live_ts >= self.ts


@dataclass(frozen=True)
class Phase:
    """A stretch of broadcast with one shot type and one thing going on."""

    start: float
    end: float
    scene: Scene
    event: Event
    side: Side
    home_score: int
    away_score: int
    lineup_home: tuple[Player, ...]
    lineup_away: tuple[Player, ...]
    ball_from: tuple[float, float]
    ball_to: tuple[float, float]
    player: Player | None = None
    graphic: bool = False
    wander: bool = True

    @property
    def duration(self) -> float:
        return max(1e-6, self.end - self.start)


def _attacking_goal(side: Side) -> tuple[float, float]:
    return (PITCH_L, PITCH_W / 2) if side is Side.HOME else (0.0, PITCH_W / 2)


def _other(side: Side) -> Side:
    return Side.AWAY if side is Side.HOME else Side.HOME


def _formation_world(side: Side, index: int) -> tuple[float, float]:
    depth, across = FORMATION[index]
    if side is Side.HOME:
        return depth * PITCH_L, across * PITCH_W
    return (1.0 - depth) * PITCH_L, (1.0 - across) * PITCH_W


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _smoothstep(u: float) -> float:
    u = _clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class MatchSim:
    """A scripted ninety minutes, or three, depending on ``duration_s``.

    Seeded throughout: the same seed gives identical frames, which is what
    makes a prompt change measurable rather than anecdotal. ``duration_s`` is
    a hard cut, so a short sim for tests and a full match for a demo are the
    same object with one number changed.
    """

    def __init__(
        self,
        *,
        seed: int = 11,
        duration_s: float = 180.0,
        clock_start_s: float = 0.0,
    ) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        self.seed = seed
        self.duration_s = duration_s
        self.clock_start_s = clock_start_s
        self.half_s = duration_s / 2.0

        rng = random.Random(seed)
        self.knowledge_pack = self._build_pack(rng)
        self._phases: list[Phase] = []
        self._ground_truth: list[GroundTruthEvent] = []
        self._whistles: list[float] = []
        self._script(rng)
        self._starts: list[float] = [p.start for p in self._phases]

        # Each player gets their own idle drift so twenty-two dots do not move
        # as one body. Drawn once here, so ``at`` stays a pure function of ts.
        jitter = random.Random(seed ^ 0x5EED)
        self._jitter: tuple[tuple[float, float, float, float], ...] = tuple(
            (
                jitter.uniform(0.35, 0.9),
                jitter.uniform(0.0, math.tau),
                jitter.uniform(0.3, 0.8),
                jitter.uniform(0.0, math.tau),
            )
            for _ in range(22)
        )

    # ---------------------------------------------------------------- notes

    def _build_pack(self, rng: random.Random) -> KnowledgePack:
        surnames = list(_SURNAMES)
        rng.shuffle(surnames)
        firsts = list(_FIRST_NAMES)
        rng.shuffle(firsts)
        cursor = 0

        def squad() -> tuple[list[Player], list[Player]]:
            nonlocal cursor
            starters: list[Player] = []
            bench: list[Player] = []
            for i, pos in enumerate(POSITIONS):
                name = f"{firsts[cursor % len(firsts)]} {surnames[cursor]}"
                cursor += 1
                starters.append(Player(name=name, number=i + 1, position=pos))
            for j, pos in enumerate(BENCH_POSITIONS):
                name = f"{firsts[cursor % len(firsts)]} {surnames[cursor]}"
                cursor += 1
                bench.append(Player(name=name, number=12 + j, position=pos))
            return starters, bench

        home_starters, home_bench = squad()
        away_starters, away_bench = squad()
        home = TeamSheet(
            name=HOME_IDENTITY[0],
            short=HOME_IDENTITY[1],
            kit=HOME_IDENTITY[2],
            formation="4-4-2",
            manager=f"{firsts[0]} {surnames[cursor]}",
            starters=home_starters,
            bench=home_bench,
        )
        away = TeamSheet(
            name=AWAY_IDENTITY[0],
            short=AWAY_IDENTITY[1],
            kit=AWAY_IDENTITY[2],
            formation="4-4-2",
            manager=f"{firsts[1]} {surnames[cursor + 1]}",
            starters=away_starters,
            bench=away_bench,
        )
        return KnowledgePack(
            home=home,
            away=away,
            competition="Continental Cup, group stage",
            venue="Ashcombe Park",
            kickoff="20:00 local",
            storylines=[
                f"{home.name} have not lost at home in eleven",
                f"{away.name} arrive on the back of three straight away wins",
                "The winner tops the group with a match to spare",
            ],
            form={home.short: "W W D W L", away.short: "W W W D W"},
            key_matchups=[
                f"{home_starters[9].surname} against {away_starters[2].surname}",
                f"{away_starters[10].surname} running at {home_starters[4].surname}",
            ],
        )

    # --------------------------------------------------------------- script

    def _script(self, rng: random.Random) -> None:
        pack = self.knowledge_pack
        lineups: dict[Side, tuple[Player, ...]] = {
            Side.HOME: tuple(pack.home.starters),
            Side.AWAY: tuple(pack.away.starters),
        }
        benches: dict[Side, list[Player]] = {
            Side.HOME: list(pack.home.bench),
            Side.AWAY: list(pack.away.bench),
        }
        score = {Side.HOME: 0, Side.AWAY: 0}
        t = 0.0
        ball = (PITCH_L / 2, PITCH_W / 2)

        def push(
            dur: float,
            scene: Scene,
            event: Event,
            side: Side,
            *,
            ball_to: tuple[float, float] | None = None,
            player: Player | None = None,
            graphic: bool = False,
            wander: bool = True,
        ) -> None:
            nonlocal t, ball
            dest = ball if ball_to is None else ball_to
            self._phases.append(
                Phase(
                    start=t,
                    end=t + dur,
                    scene=scene,
                    event=event,
                    side=side,
                    home_score=score[Side.HOME],
                    away_score=score[Side.AWAY],
                    lineup_home=lineups[Side.HOME],
                    lineup_away=lineups[Side.AWAY],
                    ball_from=ball,
                    ball_to=dest,
                    player=player,
                    graphic=graphic,
                    wander=wander,
                )
            )
            t += dur
            ball = dest

        def record(event: Event, side: Side, player: Player | None = None) -> None:
            self._ground_truth.append(
                GroundTruthEvent(
                    video_ts=round(t, 3),
                    event=event,
                    side=side,
                    player=None if player is None else player.name,
                    home_score=score[Side.HOME],
                    away_score=score[Side.AWAY],
                )
            )

        def attackers(side: Side) -> list[Player]:
            return list(lineups[side][5:])

        def defenders(side: Side) -> list[Player]:
            return list(lineups[side][1:5])

        def kick_off(side: Side) -> None:
            nonlocal ball
            self._whistles.append(t)
            record(Event.KICKOFF, side)
            ball = (PITCH_L / 2, PITCH_W / 2)
            goal = _attacking_goal(side)
            push(
                2.5,
                Scene.LIVE_PLAY,
                Event.KICKOFF,
                side,
                ball_to=(
                    PITCH_L / 2 + (goal[0] - PITCH_L / 2) * 0.12,
                    rng.uniform(20.0, 48.0),
                ),
            )

        possession = Side.HOME if rng.random() < 0.5 else Side.AWAY
        kick_off(possession)

        goals_wanted = rng.randint(2, 4)
        goal_times = sorted(rng.uniform(0.10, 0.78) * self.duration_s for _ in range(goals_wanted))
        goal_cursor = 0
        # Left to chance a short match often has no substitution at all, and
        # then nothing ever puts a name graphic on screen. Schedule two.
        sub_times = sorted(rng.uniform(0.32, 0.80) * self.duration_s for _ in range(2))
        sub_cursor = 0

        while t < self.duration_s:
            goal_mouth = _attacking_goal(possession)
            forward = 0.35 + rng.random() * 0.4
            target = (
                _clamp(ball[0] + (goal_mouth[0] - ball[0]) * forward, 6.0, PITCH_L - 6.0),
                _clamp(rng.uniform(10.0, PITCH_W - 10.0), 6.0, PITCH_W - 6.0),
            )
            hold = rng.uniform(5.0, 10.0)
            push(hold, Scene.LIVE_PLAY, Event.BUILD_UP, possession, ball_to=target)

            due_goal = goal_cursor < len(goal_times) and t >= goal_times[goal_cursor]
            due_sub = sub_cursor < len(sub_times) and t >= sub_times[sub_cursor]
            if due_goal:
                outcome = "goal"
            elif due_sub:
                outcome = "sub"
                sub_cursor += 1
            else:
                outcome = rng.choices(
                    ("shot", "corner", "foul", "throw_in", "offside", "sub"),
                    weights=(26, 17, 22, 14, 8, 13),
                )[0]

            if outcome == "goal":
                goal_cursor += 1
                shooter = rng.choice(attackers(possession))
                push(
                    1.5,
                    Scene.LIVE_PLAY,
                    Event.SHOT,
                    possession,
                    ball_to=(goal_mouth[0], PITCH_W / 2 + rng.uniform(-3.0, 3.0)),
                    player=shooter,
                    wander=False,
                )
                record(Event.SHOT, possession, shooter)
                score[possession] += 1
                record(Event.GOAL, possession, shooter)
                push(4.5, Scene.CLOSE_UP, Event.GOAL, possession, player=shooter, graphic=True)
                push(5.0, Scene.REPLAY, Event.GOAL, possession, player=shooter, wander=False)
                possession = _other(possession)
                kick_off(possession)

            elif outcome == "shot":
                shooter = rng.choice(attackers(possession))
                keeper = lineups[_other(possession)][0]
                push(
                    1.5,
                    Scene.LIVE_PLAY,
                    Event.SHOT,
                    possession,
                    ball_to=(goal_mouth[0] - 2.0, PITCH_W / 2 + rng.uniform(-6.0, 6.0)),
                    player=shooter,
                    wander=False,
                )
                record(Event.SHOT, possession, shooter)
                if rng.random() < 0.65:
                    record(Event.SAVE, _other(possession), keeper)
                    push(2.2, Scene.LIVE_PLAY, Event.SAVE, _other(possession), player=keeper)
                    push(3.5, Scene.REPLAY, Event.SAVE, _other(possession), wander=False)
                    record(Event.CORNER, possession)
                    corner_y = 0.0 if rng.random() < 0.5 else PITCH_W
                    push(
                        3.0,
                        Scene.STOPPAGE,
                        Event.CORNER,
                        possession,
                        ball_to=(goal_mouth[0], corner_y),
                    )
                    push(
                        3.5,
                        Scene.LIVE_PLAY,
                        Event.CORNER,
                        possession,
                        ball_to=(
                            goal_mouth[0] + (PITCH_L / 2 - goal_mouth[0]) * 0.12,
                            PITCH_W / 2 + rng.uniform(-8.0, 8.0),
                        ),
                    )
                else:
                    possession = _other(possession)
                    push(
                        3.0,
                        Scene.STOPPAGE,
                        Event.FREE_KICK,
                        possession,
                        ball_to=(
                            _attacking_goal(_other(possession))[0],
                            PITCH_W / 2 + rng.uniform(-8.0, 8.0),
                        ),
                    )

            elif outcome == "corner":
                record(Event.CORNER, possession)
                corner_y = 0.0 if rng.random() < 0.5 else PITCH_W
                push(
                    3.0,
                    Scene.STOPPAGE,
                    Event.CORNER,
                    possession,
                    ball_to=(goal_mouth[0], corner_y),
                )
                push(
                    3.5,
                    Scene.LIVE_PLAY,
                    Event.CORNER,
                    possession,
                    ball_to=(goal_mouth[0] * 0.88 + 6.0, PITCH_W / 2 + rng.uniform(-9.0, 9.0)),
                )
                if rng.random() < 0.5:
                    possession = _other(possession)

            elif outcome == "foul":
                offender = rng.choice(defenders(_other(possession)))
                self._whistles.append(t)
                record(Event.FOUL, _other(possession), offender)
                push(2.5, Scene.STOPPAGE, Event.FOUL, possession)
                if rng.random() < 0.35:
                    record(Event.CARD, _other(possession), offender)
                    push(
                        2.5,
                        Scene.CLOSE_UP,
                        Event.CARD,
                        _other(possession),
                        player=offender,
                        graphic=True,
                    )
                    push(
                        2.5,
                        Scene.GRAPHIC,
                        Event.CARD,
                        _other(possession),
                        player=offender,
                        graphic=True,
                    )
                push(3.0, Scene.LIVE_PLAY, Event.FREE_KICK, possession)

            elif outcome == "throw_in":
                record(Event.THROW_IN, possession)
                push(
                    2.0,
                    Scene.LIVE_PLAY,
                    Event.THROW_IN,
                    possession,
                    ball_to=(ball[0], 0.5 if ball[1] < PITCH_W / 2 else PITCH_W - 0.5),
                )
                push(2.5, Scene.LIVE_PLAY, Event.BUILD_UP, possession)

            elif outcome == "offside":
                self._whistles.append(t)
                record(Event.OFFSIDE, possession)
                push(2.5, Scene.STOPPAGE, Event.OFFSIDE, _other(possession))
                possession = _other(possession)
                push(2.5, Scene.LIVE_PLAY, Event.FREE_KICK, possession)

            else:
                side = possession if rng.random() < 0.5 else _other(possession)
                # Outfield only: a keeper coming on for a midfielder would be
                # the one thing in this match nobody would believe.
                outfield = [i for i, p in enumerate(benches[side]) if p.position != "GK"]
                if outfield:
                    coming_on = benches[side].pop(outfield[0])
                    seat = rng.randrange(5, 11)
                    lineups[side] = tuple(
                        coming_on if i == seat else p for i, p in enumerate(lineups[side])
                    )
                    record(Event.SUBSTITUTION, side, coming_on)
                    push(
                        3.5,
                        Scene.STOPPAGE,
                        Event.SUBSTITUTION,
                        side,
                        player=coming_on,
                        graphic=True,
                    )
                else:
                    push(2.5, Scene.LIVE_PLAY, Event.BUILD_UP, possession)

            if rng.random() < 0.18:
                possession = _other(possession)

        self._whistles.extend((self.half_s, self.duration_s))
        self._whistles = sorted(w for w in self._whistles if 0.0 <= w <= self.duration_s)
        self._truncate()

    def _truncate(self) -> None:
        """Cut the script at ``duration_s`` so the window is exactly as asked."""
        kept: list[Phase] = []
        for p in self._phases:
            if p.start >= self.duration_s:
                break
            if p.end > self.duration_s:
                kept.append(
                    Phase(
                        start=p.start,
                        end=self.duration_s,
                        scene=p.scene,
                        event=p.event,
                        side=p.side,
                        home_score=p.home_score,
                        away_score=p.away_score,
                        lineup_home=p.lineup_home,
                        lineup_away=p.lineup_away,
                        ball_from=p.ball_from,
                        ball_to=p.ball_to,
                        player=p.player,
                        graphic=p.graphic,
                        wander=p.wander,
                    )
                )
            else:
                kept.append(p)
        self._phases = kept
        self._ground_truth = [g for g in self._ground_truth if g.video_ts <= self.duration_s]

    # ------------------------------------------------------------- reading

    @property
    def phases(self) -> list[Phase]:
        return list(self._phases)

    @property
    def ground_truth(self) -> list[GroundTruthEvent]:
        """What actually happened. For grading only; no agent may see this."""
        return list(self._ground_truth)

    @property
    def whistle_times(self) -> list[float]:
        """When the referee blows: kickoffs, fouls, offsides, half and full time."""
        return list(self._whistles)

    @property
    def roster_names(self) -> frozenset[str]:
        pack = self.knowledge_pack
        return frozenset(p.name for p in [*pack.home.squad, *pack.away.squad])

    def phase_at(self, ts: float) -> Phase:
        ts = _clamp(ts, 0.0, self.duration_s)
        i = bisect_right(self._starts, ts) - 1
        return self._phases[max(0, min(i, len(self._phases) - 1))]

    def events_near(self, ts: float, window_s: float = 2.0) -> list[GroundTruthEvent]:
        """Ground truth within ``window_s`` before ``ts``, oldest first."""
        return [g for g in self._ground_truth if ts - window_s <= g.video_ts <= ts]

    def outcome_at(
        self,
        ts: float,
        *,
        horizon_s: float = PENDING_HORIZON_S,
        visible_after_s: float = OUTCOME_VISIBLE_AFTER_S,
    ) -> Outcome:
        """How the move that is in the balance at ``ts`` actually ends.

        This is the fact the delay buffer exists to buy. A caller watching a
        shot leave a boot does not know whether it is a goal until the picture
        moves on, and the whole delay experiment is the claim that letting it
        see those next seconds is worth the lag.

        Two numbers here are modelling assumptions rather than measurements,
        which is why they are arguments. ``horizon_s`` is how far ahead of a
        decisive moment play counts as unresolved, and ``visible_after_s`` is
        how long after the ball crosses the line the picture makes that plain,
        since the single frame it happens in is ambiguous to anybody.
        """
        phase = self.phase_at(ts)
        for g in self._ground_truth:
            if g.video_ts <= ts:
                continue
            if g.video_ts - ts > horizon_s:
                break
            if g.event in DECIDING:
                return Outcome(event=g.event, ts=g.video_ts + visible_after_s, pending=True)
        if phase.event is Event.SHOT:
            # A shot that comes to nothing still has to be called before it
            # comes to nothing, which is the failure nobody counts.
            return Outcome(event=Event.NONE, ts=phase.end + visible_after_s, pending=True)
        return Outcome(event=Event.NONE, ts=ts, pending=False)

    def score_at(self, ts: float) -> tuple[int, int]:
        p = self.phase_at(ts)
        return p.home_score, p.away_score

    def clock_at(self, ts: float) -> str:
        total = int(self.clock_start_s + _clamp(ts, 0.0, self.duration_s))
        return f"{total // 60:02d}:{total % 60:02d}"

    def at(self, ts: float) -> SimState:
        """Everything true at video time ``ts``."""
        ts = _clamp(ts, 0.0, self.duration_s)
        phase = self.phase_at(ts)
        u = (ts - phase.start) / phase.duration

        ball = self._ball_at(phase, u, ts)
        players = self._players_at(phase, ball, ts)

        focus = ball
        zoom = _ZOOM[phase.scene]
        if phase.scene is Scene.CLOSE_UP and phase.player is not None:
            for d in players:
                if d.name == phase.player.name:
                    focus = (d.x, d.y)
                    break
        zoom *= 1.0 + 0.03 * math.sin(0.31 * ts)

        return SimState(
            ts=ts,
            scene=phase.scene,
            event=phase.event,
            period=1 if ts < self.half_s else 2,
            clock=self.clock_at(ts),
            clock_s=self.clock_start_s + ts,
            home_score=phase.home_score,
            away_score=phase.away_score,
            possession=phase.side,
            ball=ball,
            players=players,
            focus=focus,
            zoom=zoom,
            graphic=phase.player if phase.graphic else None,
            graphic_side=phase.side if phase.graphic else Side.UNKNOWN,
        )

    def _ball_at(self, phase: Phase, u: float, ts: float) -> tuple[float, float]:
        eased = u if phase.wander else _smoothstep(u)
        x = phase.ball_from[0] + (phase.ball_to[0] - phase.ball_from[0]) * eased
        y = phase.ball_from[1] + (phase.ball_to[1] - phase.ball_from[1]) * eased
        if phase.wander:
            x += 2.2 * math.sin(1.9 * ts + phase.start)
            y += 1.7 * math.cos(1.4 * ts + phase.start * 0.7)
        return _clamp(x, 0.0, PITCH_L), _clamp(y, 0.0, PITCH_W)

    def _players_at(self, phase: Phase, ball: tuple[float, float], ts: float) -> tuple[Dot, ...]:
        dots: list[Dot] = []
        for side, lineup in ((Side.HOME, phase.lineup_home), (Side.AWAY, phase.lineup_away)):
            attacking = side is phase.side
            for i, player in enumerate(lineup):
                bx, by = _formation_world(side, i)
                jx, jpx, jy, jpy = self._jitter[len(dots)]
                if i == 0:
                    # The keeper holds the line of their own goal and only
                    # shuffles across as the ball moves.
                    x = 3.0 if side is Side.HOME else PITCH_L - 3.0
                    y = PITCH_W / 2 + (ball[1] - PITCH_W / 2) * 0.35
                else:
                    drift = (ball[0] - PITCH_L / 2) * (0.55 if attacking else 0.45)
                    x = bx + drift
                    y = by
                    pull = 0.5 * math.exp(-math.hypot(x - ball[0], y - ball[1]) / 22.0)
                    x += (ball[0] - x) * pull
                    y += (ball[1] - y) * pull
                    x += 1.5 * math.sin(jx * ts + jpx)
                    y += 1.2 * math.cos(jy * ts + jpy)
                dots.append(
                    Dot(
                        side=side,
                        number=player.number or (i + 1),
                        name=player.name,
                        x=_clamp(x, 0.5, PITCH_L - 0.5),
                        y=_clamp(y, 0.5, PITCH_W - 0.5),
                    )
                )
        return tuple(dots)


_ZOOM: dict[Scene, float] = {
    Scene.LIVE_PLAY: 1.0,
    Scene.STOPPAGE: 1.15,
    Scene.REPLAY: 1.35,
    Scene.CLOSE_UP: 3.4,
    Scene.CROWD: 1.2,
    Scene.GRAPHIC: 1.0,
}
