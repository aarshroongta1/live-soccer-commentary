"""The fact gate: the last thing between a model's sentence and a microphone.

The claim this project makes is that a vision model can call a match without
inventing things, and this module is where that claim is cashed. It is
deterministic and it makes no model call, because a checker that hallucinates
is not a checker. Every rejection carries a short tag plus the detail, so the
eval can report rejection rate broken down by reason and a human can grep a
90-minute trace for the one line that got through.

Four rules, in the order a sceptic would apply them.

* A replay is never called as live. Narrating a replay as though it were
  happening is the most embarrassing failure available to this system.
* Names must be on a roster in the knowledge pack. A name the caller claims to
  have read off a graphic gets *less* latitude, not more: if it is not on a
  roster then the caller did not read a graphic, it imagined one, and the rest
  of that line is suspect with it.
* A scoreline said out loud must match the board.
* A goal is only a goal once the board says so or the lookahead frames show a
  celebration. Nothing else gets to claim one. "The board says so" is the
  caller's own runtime's judgement and it is broader than a settled change:
  a board part-way through agreeing counts, and so does a goal the state has
  already taken in, which is what every line *about* a goal comes after.

Where a line can be saved it is saved. Dropping a name the gate cannot verify
and leaving "the cross comes in and the winger cuts inside" is better
commentary than silence, so an unverifiable name in the body of a line is
trimmed rather than fatal — unless trimming leaves too little to be worth
saying.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from commentary.config import SETTINGS, GateConfig
from commentary.schemas import CallerLine, Event, GateVerdict, KnowledgePack, MatchState, Scene
from commentary.state import parse_sighting

# Words that open sentences or describe football, not people. A capitalised
# token in here is never treated as a name, which is what stops the gate
# trimming "Brilliant from the far post" down to "from the far post".
_STOPWORD_TEXT = """
    a an and as at back but by for from in into of off on onto or out over the then to
    towards under up with within without
    i he she it they we you him her them his hers its their our your me my mine
    that this these those there here what when where which who whom whose why how
    all any both each every few more most no none not now nor only other same some such
    still too very yes oh ah well just again almost nearly already
    is am are was were be been being do does did done has have had having
    can could may might must shall should will would let
    goal goals ball cross corner corners penalty penalties header headers keeper goalkeeper
    referee ref offside card cards yellow red foul fouls throw box area pitch half time
    full whistle replay touch pass passes shot shots save saves free kick kicks run runs
    side left right centre center middle midfield defence defense attack striker winger
    captain sub subs substitution bench minute minutes second seconds injury stoppage var
    play played plays playing goes going get gets got take takes taken drive drives driven
    win wins won lose loses lost find finds found send sends sent turn turns turned
    break breaks broken clip clips whip whips cut cuts chip chips slot slots
    look looks looking wait waits try tries come comes coming go gone
    great good brilliant lovely superb terrific poor quick sharp clever strong big small
    late early first third final last long short high low deep wide narrow tight
    another one two three four five six seven eight nine ten nil nought zero level
    away home down forward forwards square above below behind beyond against between
    inside outside around along across straight
    after before while since until than so if because though although unless even never
    always surely much nice easy lucky unlucky dangerous clear close magnificent outstanding
    fine smart neat tidy calm cool sweet huge massive well ooh wow yeah okay
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

#: Prepositions left dangling by a trim ("comes in from  and the winger"), so
#: they go with the name rather than staying behind as debris.
_DANGLERS = "from|by|to|for|off|with|of|onto|into|at|on|through|past"

_WORD = re.compile(r"[^\W\d_]+(?:['’‐-][^\W\d_]+)*", re.UNICODE)
_NUMBER_WORDS: dict[str, int] = {
    "nil": 0,
    "nought": 0,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD_ALT = "|".join(_NUMBER_WORDS)
_DIGIT_PAIR = re.compile(r"\b([0-9])\s*[-–—:]\s*([0-9])\b")
_WORD_PAIR = re.compile(rf"\b({_WORD_ALT})[\s-]+({_WORD_ALT})\b", re.IGNORECASE)
_ALL_PAIR = re.compile(rf"\b({_WORD_ALT})[\s-]+all\b", re.IGNORECASE)

#: How a line says a goal was scored. The word "goal" is not on the list.
#: It used to be, with a second expression stripping the innocent uses first
#: — "towards the goal", "goal kick", "goalkeeper" — and that list can never
#: be finished: the first real run lost "France scrambling back towards their
#: own goal" to it, and "his goal" and "the French goal" were next. The form
#: already carries an unambiguous answer in ``event``, so the prose only has
#: to catch the phrasings that mean a goal and nothing else.
_GOAL_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bscores\b(?!\s+(?:are|level|tied))",
        r"\bscored\b",
        r"\bit'?s\s+in\b",
        r"\b(?:in|into)\s+the\s+(?:back\s+of\s+the\s+)?net\b",
        r"\bfinds?\s+the\s+net\b",
        r"\bmakes?\s+it\s+\d\b",
        r"\bequalis\w*\b",
        r"\bequaliz\w*\b",
    )
)


def fold(text: str) -> str:
    """Accents off, case off, punctuation to spaces.

    The caller reads names off a broadcast graphic and the roster comes off a
    web page; the two agree on the person and disagree on the diacritics far
    more often than is comfortable, so neither is compared in its own spelling.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    plain = plain.replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^0-9A-Za-z]+", " ", plain).lower().split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _surname(name: str) -> str:
    return name.rsplit(" ", 1)[-1]


@dataclass(frozen=True)
class _Roster:
    """Every string the gate is willing to let a line say out loud."""

    people: frozenset[str]
    teams: frozenset[str]
    numbers: frozenset[str]

    @property
    def known(self) -> frozenset[str]:
        return self.people | self.teams


def _roster_of(state: MatchState, pack: KnowledgePack | None) -> _Roster:
    """Fold the knowledge pack and the match state into one bag of allowed names.

    The state contributes the two team names and whatever the graphics have
    taught it about shirt numbers; the pack contributes the squads, the
    managers, the competition and the ground.
    """
    people: set[str] = set()
    teams: set[str] = set()
    numbers: set[str] = set()

    for team_name in (state.home, state.away):
        folded = fold(team_name)
        teams.add(folded)
        teams.update(token for token in folded.split() if len(token) >= 3)
    for number, name in state.on_pitch.items():
        numbers.add(fold(number))
        people.add(fold(name))

    if pack is not None:
        for sheet in (pack.home, pack.away):
            for label in (sheet.name, sheet.short):
                if label:
                    teams.add(fold(label))
                    teams.update(token for token in fold(label).split() if len(token) >= 3)
            if sheet.manager:
                people.add(fold(sheet.manager))
                people.add(fold(_surname(sheet.manager)))
            for player in sheet.squad:
                people.add(fold(player.name))
                people.add(fold(player.surname))
                people.update(token for token in fold(player.name).split() if len(token) >= 3)
                if player.number is not None:
                    numbers.add(str(player.number))
        for label in (pack.competition, pack.venue):
            teams.update(token for token in fold(label).split() if len(token) >= 4)

    people.discard("")
    teams.discard("")
    return _Roster(frozenset(people), frozenset(teams), frozenset(numbers))


@dataclass(frozen=True)
class _Candidate:
    """A capitalised run in the line that looks like it names somebody."""

    text: str
    start: int
    end: int


def _candidates(line: str) -> list[_Candidate]:
    """Consecutive capitalised words, minus the ones that are just English.

    Runs rather than single words, so "Jude Bellingham" is checked against the
    roster as one person instead of as two unknown halves.
    """
    runs: list[_Candidate] = []
    current: list[re.Match[str]] = []

    def flush() -> None:
        while current and fold(current[0].group()) in _STOPWORDS:
            current.pop(0)
        while current and fold(current[-1].group()) in _STOPWORDS:
            current.pop()
        if current:
            runs.append(
                _Candidate(
                    line[current[0].start() : current[-1].end()],
                    current[0].start(),
                    current[-1].end(),
                )
            )
        current.clear()

    previous_end = -1
    for match in _WORD.finditer(line):
        word = match.group()
        adjacent = previous_end >= 0 and match.start() - previous_end <= 1
        if len(word) > 1 and word[0].isupper():
            if not adjacent:
                flush()
            current.append(match)
        else:
            flush()
        previous_end = match.end()
    flush()
    return [c for c in runs if fold(c.text) not in _STOPWORDS]


def _matches_roster(name: str, roster: _Roster, threshold: float) -> bool:
    """Exact after folding, or close enough that it is the same person misspelt.

    The suffix rule is the one that earns its place. ``Player.surname`` is a
    split on the last space, so Di María's surname is "María", De Paul's is
    "Paul" and Mac Allister's is "Allister"; the roster's token list drops
    anything under three letters, so "di" and "de" are never known; and the
    similarity of "di maria" to "angel di maria" is 0.73, well under the
    threshold. Every compound surname therefore failed — five lines of one
    Sonnet run died on exactly this. A candidate that ends a known full name
    on a word boundary is that person, and nothing else is.
    """
    folded = fold(name)
    if not folded:
        return True
    known = roster.known
    if folded in known:
        return True
    parts = folded.split()
    if len(parts) > 1 and all(part in known for part in parts):
        return True
    suffix = f" {folded}"
    if any(entry.endswith(suffix) for entry in known):
        return True
    return any(_similar(folded, entry) >= threshold for entry in known)


def _stated_scores(line: str) -> list[tuple[int, int]]:
    """Every scoreline the line says out loud, in digits or in words."""
    found: list[tuple[int, int]] = []
    for match in _DIGIT_PAIR.finditer(line):
        found.append((int(match.group(1)), int(match.group(2))))
    for match in _WORD_PAIR.finditer(line):
        first, second = match.group(1).lower(), match.group(2).lower()
        if (first, second) == ("one", "two"):
            continue  # "a lovely one-two" is a give-and-go, not a scoreline.
        found.append((_NUMBER_WORDS[first], _NUMBER_WORDS[second]))
    for match in _ALL_PAIR.finditer(line):
        value = _NUMBER_WORDS[match.group(1).lower()]
        found.append((value, value))
    return found


def _claims_goal(line: CallerLine) -> bool:
    """Whether this line asserts that a goal has been scored, form or prose.

    The form field is the half that is not guesswork: a caller describing a
    goal marks the event as one. The phrases are for the line that says it
    without the form agreeing.
    """
    if line.event is Event.GOAL:
        return True
    return any(pattern.search(line.line) for pattern in _GOAL_CLAIMS)


def _tidy(text: str) -> str:
    """Repair a line that has had a name cut out of the middle of it."""
    out = re.sub(rf"\b(?:{_DANGLERS})\s+(?=(?:[,.;!?]|and\b|as\b|but\b|who\b|then\b|$))", "", text)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)
    out = re.sub(r"^[\s,;:.!?-]+", "", out)
    out = re.sub(r"^(?:and|but|as)\s+", "", out, flags=re.IGNORECASE)
    out = out.strip()
    return out[:1].upper() + out[1:] if out else out


def _word_count(text: str) -> int:
    return len([word for word in text.split() if any(ch.isalnum() for ch in word)])


@dataclass
class GateStats:
    """Running totals, so a run can print the table without re-deriving it."""

    judged: int = 0
    passed: int = 0
    trimmed: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return self.judged - self.passed

    def record(self, verdict: GateVerdict) -> None:
        self.judged += 1
        tags = [reason.split(":", 1)[0] for reason in verdict.reasons]
        if verdict.passed:
            self.passed += 1
            if "trimmed_name" in tags:
                self.trimmed += 1
            return
        for tag in tags:
            if tag != "trimmed_name":
                self.by_reason[tag] += 1

    def table(self) -> str:
        header = (
            f"judged {self.judged}  passed {self.passed} ({self.trimmed} trimmed)  "
            f"rejected {self.rejected}"
        )
        rows = [f"  {tag:<28} {count}" for tag, count in self.by_reason.most_common()]
        return "\n".join([header, *rows])


class FactGate:
    """Judges one caller line against the roster, the board, and the picture."""

    def __init__(self, cfg: GateConfig = SETTINGS.gate) -> None:
        self.cfg = cfg
        self.stats = GateStats()

    def judge(
        self,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        board_changed: bool = False,
        lookahead_celebration: bool = False,
    ) -> GateVerdict:
        """Pass, trim, or reject — and always say why.

        ``board_changed`` is whether the scoreboard supports a goal claimed
        at this moment: a settled change near the cursor, a change the board
        reader is still confirming, or a goal already in the state.
        """
        verdict = self._judge(
            line,
            state,
            pack,
            board_changed=board_changed,
            lookahead_celebration=lookahead_celebration,
        )
        self.stats.record(verdict)
        return verdict

    def _judge(
        self,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        board_changed: bool,
        lookahead_celebration: bool,
    ) -> GateVerdict:
        text = line.line.strip()
        if line.scene is Scene.REPLAY:
            return GateVerdict(
                passed=False, reasons=["scene_replay: a replay is never called as live"]
            )
        if not line.speak:
            return GateVerdict(passed=False, reasons=["not_speaking: caller chose silence"])
        if not text:
            return GateVerdict(passed=False, reasons=["empty_line: nothing to say"])

        roster = _roster_of(state, pack)
        fatal: list[str] = []
        fatal += self._check_names_read(line, roster, pack)
        fatal += self._check_scoreline(text, state)
        if (
            self.cfg.require_board_for_goal
            and _claims_goal(line)
            and not (board_changed or lookahead_celebration)
        ):
            fatal.append("unconfirmed_goal: no board change and no celebration in the lookahead")
        if fatal:
            return GateVerdict(passed=False, reasons=fatal)

        return self._trim_unverified(text, roster)

    def _check_names_read(
        self, line: CallerLine, roster: _Roster, pack: KnowledgePack | None
    ) -> list[str]:
        """A name claimed off a graphic is held to the roster like any other.

        This is the one place the gate refuses to trim. A name in ``names_read``
        that is on no roster means the caller did not misjudge a face, it
        invented a graphic, and a line built on an invented graphic is not
        worth saving.

        A sighting is read the way the state reads it before it is checked.
        The caller writes what it saw — "11 Di María", "Di María (11)" —
        because that is the pairing that justifies the name, and matching the
        whole string against the roster rejected every one of them. Parsed,
        it is two claims and both are checked: the number is in the squad and
        the name is on the roster.
        """
        problems: list[str] = []
        for read in line.names_read:
            token = read.strip()
            if not token:
                continue
            sighting = parse_sighting(token)
            if sighting is not None:
                number, name = sighting
                problems += self._check_number(str(number), roster, pack)
                if not _matches_roster(name, roster, self.cfg.name_match_threshold):
                    problems.append(f"name_read_not_on_roster: {name}")
                continue
            if token.isdigit():
                problems += self._check_number(token, roster, pack)
                continue
            if not _matches_roster(token, roster, self.cfg.name_match_threshold):
                problems.append(f"name_read_not_on_roster: {token}")
        return problems

    def _check_number(
        self, number: str, roster: _Roster, pack: KnowledgePack | None
    ) -> list[str]:
        if pack is None:
            return []
        squad = {known.lstrip("0") for known in roster.numbers}
        if number.lstrip("0") in squad:
            return []
        return [f"number_not_in_squad: {number}"]

    def _check_scoreline(self, text: str, state: MatchState) -> list[str]:
        """Either orientation is allowed: the gate cannot know which team was meant."""
        board = (state.home_score, state.away_score)
        problems: list[str] = []
        for home, away in _stated_scores(text):
            if (home, away) != board and (away, home) != board:
                problems.append(
                    f"scoreline_mismatch: said {home}-{away}, board {board[0]}-{board[1]}"
                )
        return problems

    def _trim_unverified(self, text: str, roster: _Roster) -> GateVerdict:
        """Cut the names that cannot be verified and keep whatever still stands up."""
        reasons: list[str] = []
        kept = text
        for candidate in sorted(_candidates(text), key=lambda c: c.start, reverse=True):
            if _matches_roster(candidate.text, roster, self.cfg.name_match_threshold):
                continue
            reasons.append(f"name_not_on_roster: {candidate.text}")
            reasons.append(f"trimmed_name: {candidate.text}")
            kept = kept[: candidate.start] + kept[candidate.end :]
        if reasons:
            kept = _tidy(kept)

        words = _word_count(kept)
        if words < self.cfg.min_words_after_trim:
            reasons.append(f"too_short_after_trim: {words} words left")
            return GateVerdict(passed=False, reasons=reasons)
        return GateVerdict(passed=True, reasons=reasons, line=kept)
