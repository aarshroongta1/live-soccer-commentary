"""Regenerate ``commentary.prompts.commentary_examples`` from real captions.

The phraser is not taught a register by adjectives. It is shown what a
commentator actually said, in the caption file a broadcaster shipped with the
video, and asked to sound like that. So the example set has to come out of
captions rather than out of anybody's idea of how football is called, and it
has to be regenerable: point this at more caption files later and the set
grows.

    uv run python scripts/build_commentary_examples.py \
        --captions clips/argfra-dimaria.en.json3 \
        --pack clips/pack-argfra-2022.json \
        --out src/commentary/prompts/commentary_examples.py

**Several matches at once.** ``--captions`` and ``--pack`` each take one or
more paths and are paired in order — the first ``--captions`` file with the
first ``--pack``, the second with the second, and so on — rather than a
combined ``captions.json3:pack.json`` token, because that pairing is already
how argparse's ``nargs="+"`` works and needed no new syntax:

    uv run python scripts/build_commentary_examples.py \
        --captions clips/argfra-dimaria.en.json3 clips/lei-mun-2015.en.json3 \
        --pack clips/pack-argfra-2022.json clips/pack-3754186.json \
        --after 300 78

Each caption file is filtered against *its own* pack's team sheet — a name
true on one match's roster is not automatically true on another's — but the
"already said in lower case somewhere" vocabulary that separates a real word
from a mangled surname (see :func:`names_are_clean`) is pooled across every
caption file given, the same as when a single match's commentary was split
across several files. ``--after`` is one kickoff offset per file, same order
as ``--captions``; give fewer values than files and the last one repeats, so
a single ``--after 300`` still applies to every file the way it always did.

What it does, in order.

1. **Utterances, not caption segments.** YouTube writes a rolling caption:
   half-sentences, stamped when they appeared on screen, with the sentence
   finishing two segments later. The stream is reassembled and split again on
   the three things that really end an utterance — a sentence-final stop, a
   ``>>`` speaker change, and a gap of two seconds or more. This is the same
   reconstruction ``runs/prompt-name/REAL_COMMENTARY.md`` measured, so the
   medians the phraser's prompt quotes are medians of these strings.

2. **Live play only, as far as captions can say.** Anything bracketed is
   crowd noise the ASR gave up on. Anything in the studio vocabulary is the
   build-up or the half-time panel. Anything before ``--after`` is neither.

3. **No garbled names.** Auto-captions mangle surnames constantly, and a
   mangled surname in a prompt is a name the phraser may repeat out loud. So
   every capitalised word that does not start the utterance has to be on the
   team sheet or in a short list of words a broadcast says. One word that is
   not drops the utterance whole — there are thousands left.

4. **A kind, by keyword.** Loose on purpose: the kinds exist so the prompt can
   show a spread rather than forty bare surnames, not so that anything
   downstream can trust the label.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from commentary.grading.captions import load_json3
from commentary.grading.transcripts import Segment
from commentary.schemas import KnowledgePack

#: A gap this long between two caption segments ends the utterance.
GAP_S = 2.0

#: Longest utterance kept. The longest thing anybody said in half an hour of
#: live football was 28 words; the character cap is what holds the generated
#: file inside the 100-column line length, and it is the one that bites.
MAX_WORDS = 28
MAX_CHARS = 88

_SPEAKER = re.compile(r">>\s*")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*$")

#: Words a broadcast says out loud that are on nobody's team sheet. Anything
#: capitalised, mid-utterance and not here or on the sheet is treated as a
#: mangled name and costs the whole utterance.
_ALLOWED_TEXT = """
var fifa world cup qatar lusail doha final finals tournament europe
america american south african european premier league champions
i i'm i've i'll i'd o oh ah ok okay yes no now here there and but so
monday tuesday wednesday thursday friday saturday sunday
"""
ALLOWED_PROPER = frozenset(_ALLOWED_TEXT.split())

#: The studio, the adverts and the panel. None of it is live play.
STUDIO = (
    "brought to you by",
    "sponsor",
    "coming up",
    "after the break",
    "welcome back",
    "welcome to",
    "join us",
    "half-time",
    "half time",
    "full-time",
    "full time",
    "team news",
    "line-ups",
    "lineups",
    "studio",
    "let's take a look",
    "highlights",
    "subscribe",
)

#: First match wins, so the order is the order of specificity.
KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "goal",
        (
            "goal",
            "scores",
            "scored",
            "it's in",
            "in the net",
            "buries",
            "back of the net",
            "equaliser",
            "equalizer",
        ),
    ),
    (
        "save",
        ("save", "saves", "saved", "keeper", "goalkeeper", "parried", "tipped", "denied", "palms"),
    ),
    (
        "shot",
        (
            "shot",
            "shoots",
            "strikes",
            "fires",
            "fired",
            "curls",
            "drives it",
            "volley",
            "header",
            "heads",
            "wide",
            "over the bar",
            "the post",
            "crossbar",
            "chance",
            "effort",
            "blocked",
        ),
    ),
    (
        "foul",
        (
            "foul",
            "fouled",
            "yellow",
            "red card",
            "card",
            "booked",
            "booking",
            "referee",
            "penalty",
            "offside",
            "handball",
            "whistle",
        ),
    ),
    (
        "dead_ball",
        (
            "corner",
            "free kick",
            "free-kick",
            "throw",
            "goal kick",
            "restart",
            "kick-off",
            "kickoff",
            "the wall",
            "the spot",
        ),
    ),
    (
        "pass",
        (
            "played by",
            "on by",
            "out by",
            "in by",
            "flicked",
            "collected",
            "regathered",
            "moved on",
            "passes",
            "pass",
            "finds",
            "square",
            "threaded",
            "cross",
            "crosses",
            "knocks",
            "knocked",
            "sifted",
            "spelled",
            "touched",
            "chased",
            "taken up",
            "carries",
            "runs",
            "away from",
            "wriggl",
            "dispossessed",
            "intercept",
            "cleared",
            "clearance",
            "tackle",
            "forward",
            "backwards",
        ),
    ),
)

#: Kinds in the order the prompt shows them.
KIND_ORDER = ("build_up", "pass", "shot", "save", "goal", "foul", "dead_ball", "aside")

#: How many of each to keep. Build-up and passing are most of live commentary
#: and most of what the caller writes badly, so they get the room.
#:
#: The loud three — goal, shot, save — are set above anything these captions
#: can supply, so every chance utterance that survives the filters is kept.
#: They were a cap before, and the goal bucket was losing seven of the
#: twenty-seven found for no better reason than a round number, while the
#: phraser was writing "Mbappé!" over a volley. There is no reason to ration
#: the examples of the thing being got wrong.
QUOTA = {
    "build_up": 60,
    "pass": 55,
    "shot": 40,
    "save": 25,
    "goal": 30,
    "foul": 35,
    "dead_ball": 30,
    "aside": 25,
}


@dataclass(frozen=True)
class Utterance:
    """One thing a commentator said, with the second it started."""

    ts: float
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def utterances(segments: Sequence[Segment]) -> Iterator[Utterance]:
    """Caption segments back into the sentences somebody actually spoke."""
    start: float | None = None
    parts: list[str] = []
    previous_end = 0.0

    for segment in segments:
        raw = segment.text
        gap = start is not None and segment.start - previous_end >= GAP_S
        if (gap or raw.lstrip().startswith(">>")) and parts and start is not None:
            yield Utterance(ts=start, text=" ".join(parts))
            parts, start = [], None
        for piece in _SPEAKER.split(raw):
            for word in piece.split():
                if start is None:
                    start = segment.start
                parts.append(word)
                if _SENTENCE_END.search(word):
                    yield Utterance(ts=start, text=" ".join(parts))
                    parts, start = [], None
        previous_end = segment.end

    if parts and start is not None:
        yield Utterance(ts=start, text=" ".join(parts))


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def roster_words(pack: KnowledgePack) -> set[str]:
    """Every word of every name on either sheet, folded to plain lower case."""
    words: set[str] = set()
    for sheet in (pack.home, pack.away):
        for label in (sheet.name, sheet.short, sheet.demonym, sheet.manager or ""):
            words.update(_letters(label).split())
        for player in sheet.squad:
            words.update(_letters(player.name).split())
    return {word for word in words if word}


def _letters(text: str) -> str:
    return re.sub(r"[^a-z' ]+", " ", strip_accents(text.lower()))


def names_are_clean(
    text: str, allowed: frozenset[str] | set[str], common: frozenset[str] | set[str]
) -> bool:
    """Is every capitalised word here a word a broadcast really says?

    Every capitalised word is a proper noun on the team sheet, a word in the
    allow-list, an ordinary word that happens to start a sentence, or the
    ASR's guess at a surname. The last is the thing this exists to keep out of
    the prompt, and telling it from the third is what ``common`` is for: the
    set of words the same captions use in lower case somewhere else. "Spread"
    opening a sentence is safe because "spread" appears lower-cased a hundred
    times; "MacAllister" opening one is not, because it never does — and the
    team sheet spells him Mac Allister.

    Without that set the first word had to be exempt, and a bare surname is
    exactly the shape of line worth having, so every mangled one got in.
    """
    for token in text.split():
        word = re.sub(r"[^A-Za-z'\-]", "", token)
        if len(word) < 2 or not word[0].isupper():
            continue
        folded = strip_accents(word.lower())
        base = folded[:-2] if folded.endswith("'s") else folded
        if folded in common or base in common:
            continue
        for part in base.replace("-", " ").split():
            if part and part not in allowed:
                return False
    return True


def kind_of(text: str) -> str:
    """A loose label, by keyword, first rule wins."""
    lowered = text.lower()
    # "Goal kick given." is a restart, and the goal rule would otherwise
    # claim it on the word "goal" before dead_ball ever gets a look.
    if "goal kick" in lowered:
        return "dead_ball"
    for kind, needles in KIND_RULES:
        if any(needle in lowered for needle in needles):
            return kind
    return "build_up" if len(text.split()) <= 4 else "aside"


def keep(
    utterance: Utterance, allowed: set[str], common: set[str], *, after_s: float
) -> bool:
    """Is this a line of live commentary with nothing invented in it?"""
    text = utterance.text
    if utterance.ts < after_s:
        return False
    if "[" in text or "]" in text or '"' in text or "\\" in text:
        return False
    if not 1 <= utterance.words <= MAX_WORDS or len(text) > MAX_CHARS:
        return False
    if not re.search(r"[a-z]", text):
        return False
    # A line that opens in lower case lost its beginning to a caption
    # boundary, and a line with no full stop never reached its end.
    if not text[0].isalnum() or text[0].islower():
        return False
    if not _SENTENCE_END.search(text):
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in STUDIO):
        return False
    words = lowered.split()
    if any(a == b for a, b in zip(words, words[1:], strict=False)):
        # "That is a a new record" — the ASR stuttering, not the commentator.
        return False
    return names_are_clean(text, allowed, common)


def lowercase_vocabulary(paths: Iterable[Path]) -> set[str]:
    """Every word these captions ever write in lower case.

    This is what separates an ordinary word opening a sentence from a surname
    the ASR invented. See :func:`names_are_clean`.
    """
    words: set[str] = set()
    for path in paths:
        for segment in load_json3(path).segments:
            for token in segment.text.split():
                word = re.sub(r"[^A-Za-z']", "", token)
                if word and word[0].islower():
                    words.add(strip_accents(word.lower()))
    return words


def spread(found: list[str], quota: int) -> list[str]:
    """``quota`` of these, drawn evenly across the match rather than off the top.

    The head of the list is the first ten minutes after kickoff. A prompt full
    of one team's opening spell is a prompt about that spell; an even stride
    over ninety minutes is a prompt about football.
    """
    if len(found) <= quota:
        return list(found)
    step = len(found) / quota
    return [found[int(i * step)] for i in range(quota)]


def surnames_of(pack: KnowledgePack) -> set[str]:
    """Every surname on either sheet, folded, for the build-up filter."""
    return {
        strip_accents(player.surname.lower())
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
    }


@dataclass(frozen=True)
class Source:
    """One caption file, the pack that says which of its names are real, and
    when its live play starts."""

    path: Path
    pack: KnowledgePack
    after_s: float


def gather(sources: Sequence[Source]) -> dict[str, list[str]]:
    """Every kept utterance, by kind, in the order each broadcast said them.

    Roster words — what makes a capitalised word a real name rather than a
    mangled one — come from each source's own pack, so a name true on one
    match's team sheet cannot wave a different match's utterance through.
    The lower-case vocabulary that backs that same check (see
    :func:`names_are_clean`) is pooled across every caption file, exactly as
    it was when one match's commentary arrived split across several files.
    """
    common = lowercase_vocabulary(source.path for source in sources)
    seen: set[str] = set()
    found: dict[str, list[str]] = {kind: [] for kind in KIND_ORDER}
    for source in sources:
        allowed = set(ALLOWED_PROPER) | roster_words(source.pack)
        surnames = surnames_of(source.pack)
        for utterance in utterances(load_json3(source.path).segments):
            if not keep(utterance, allowed, common, after_s=source.after_s):
                continue
            key = utterance.text.lower()
            if key in seen:
                continue
            kind = kind_of(utterance.text)
            # Build-up has no keyword of its own — it is whatever matched
            # nothing else and stayed short — so without a name in it the
            # bucket fills with clock-watching and crowd noise. Real build-up
            # commentary *is* the names: "De Paul." "Now Di María."
            if kind == "build_up" and not _has_surname(key, surnames):
                continue
            seen.add(key)
            found[kind].append(utterance.text)
    return found


def _has_surname(folded_text: str, surnames: set[str]) -> bool:
    words = {re.sub(r"[^a-z']", "", word) for word in strip_accents(folded_text).split()}
    return bool(words & surnames)


def render(found: dict[str, list[str]], sources: Sequence[Path]) -> str:
    """The generated module, byte-stable for a given input."""
    chosen = {kind: spread(found[kind], QUOTA[kind]) for kind in KIND_ORDER}
    total = sum(len(v) for v in chosen.values())
    plural = "s" if len(sources) > 1 else ""
    listed = "\n".join(f"  {path}" for path in sources)
    head = f'''"""Real commentary, as a broadcaster's own captions recorded it.

Generated by ``scripts/build_commentary_examples.py``. Do not edit by hand:
run the script again, against more caption files if there are any.

These are not illustrations of a style. They are {total} things a professional
actually said during live play, reassembled into utterances and filtered down
to the ones whose every name is on a team sheet — an auto-caption mangles a
surname about as often as it gets one right, and a mangled surname in a
prompt is a name the phraser may say out loud.

The kinds are keyword guesses and nothing downstream should trust them. They
exist so the prompt can show a spread, a few of each, instead of forty bare
surnames in a row, which is what an unsorted sample of live commentary is.

Source{plural}:
{listed}
"""

from __future__ import annotations

#: Utterances by kind, in the order the prompt shows them.
EXAMPLES: dict[str, tuple[str, ...]] = {{
'''
    body: list[str] = []
    for kind in KIND_ORDER:
        body.append(f'    "{kind}": (')
        body.extend(f'        "{text}",' for text in chosen[kind])
        body.append("    ),")
    tail = '''}

#: The kinds, in prompt order.
KINDS: tuple[str, ...] = tuple(EXAMPLES)
'''
    return head + "\n".join(body) + "\n" + tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rebuild the phraser's example set")
    parser.add_argument(
        "--captions",
        nargs="+",
        default=["clips/argfra-dimaria.en.json3"],
        help="yt-dlp .en.json3 caption files",
    )
    parser.add_argument(
        "--pack",
        nargs="+",
        default=["clips/pack-argfra-2022.json"],
        help="one knowledge pack per --captions file, same order: each file's own pack "
        "says which of its capitalised words are real names",
    )
    parser.add_argument(
        "--after",
        type=float,
        nargs="+",
        default=[300.0],
        help="one kickoff offset per --captions file, same order — the build-up is not "
        "live play; give fewer values than files and the last one repeats",
    )
    parser.add_argument("--out", default="src/commentary/prompts/commentary_examples.py")
    parser.add_argument("--dry-run", action="store_true", help="print the tally and stop")
    args = parser.parse_args(argv)

    captions = [Path(p) for p in args.captions]
    pack_paths = [Path(p) for p in args.pack]
    if len(pack_paths) != len(captions):
        parser.error(
            f"--pack must give one pack per --captions file: "
            f"{len(captions)} captions, {len(pack_paths)} packs"
        )
    afters = list(args.after)
    if len(afters) > len(captions):
        parser.error(
            f"--after gives more values ({len(afters)}) than --captions files ({len(captions)})"
        )
    if len(afters) < len(captions):
        afters += [afters[-1]] * (len(captions) - len(afters))

    sources = [
        Source(
            path=cap,
            pack=KnowledgePack.model_validate(json.loads(pack_path.read_text(encoding="utf-8"))),
            after_s=after,
        )
        for cap, pack_path, after in zip(captions, pack_paths, afters, strict=True)
    ]

    found = gather(sources)

    total = 0
    for kind in KIND_ORDER:
        kept = len(spread(found[kind], QUOTA[kind]))
        total += kept
        print(f"{kind:<10} kept {kept:>3} of {len(found[kind]):>4} found")
    print(f"total {total}")
    if args.dry_run:
        return 0

    out = Path(args.out)
    out.write_text(render(found, captions), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
