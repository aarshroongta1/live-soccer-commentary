"""The four reference numbers the corpus study does not state, measured.

``docs/research/real-commentary-corpus.md`` gives most of what
:mod:`commentary.grading.register_reference` holds. Four of its bands are not
in the document — two words or fewer, gaps over eight seconds, numbers off the
scoreline, and a repeated opener — and this is where those values come from,
so that "measured here" in that module's ``basis`` has a file behind it.

    uv run python scripts/corpus_numbers.py

The method is the study's own (its "Method, in one paragraph"): the caption
stream is reassembled into utterances by
:func:`scripts.build_commentary_examples.utterances`, bracketed non-speech is
stripped, and only the live window of study section 1 is counted. The proof
that it is the same method is the first thing printed — the pooled club row
comes back as 6,378 utterances and 67,993 words with a median of 8, mean 10.7,
4.0% of one word, 23.3% of four or fewer, 47.3% of nine or more and 20.2% of
sixteen or more, which is section 1's table exactly, and a pooled gap median
of 4.3 s with 53% over four seconds, which is section 2.2's.

The number and opener figures are computed with the very regexes
:mod:`commentary.grading.register` applies to a trace, and the bare-name and
name shares with its own predicates against each match's team sheet, so the
reference and the measurement are the same instrument pointed two ways.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from build_commentary_examples import utterances  # noqa: E402

from commentary.gate import fold  # noqa: E402
from commentary.grading.captions import load_json3  # noqa: E402
from commentary.grading.register import (  # noqa: E402
    NUMBER,
    OPENER_WINDOW,
    SCORELINE_SAID,
    is_bare_name,
    name_words,
    says_a_name,
)
from commentary.schemas import KnowledgePack  # noqa: E402

_BRACKET = re.compile(r"\[[^\]]*\]")

#: Each caption file, its live window from study section 1, and the pack
#: whose team sheet a name is checked against.
CORPUS: tuple[tuple[str, float, float, str], ...] = (
    ("pl-liv-mun-2025", 152, 6110, "clips/pack-pl-liv-mun-2025.json"),
    ("pl-tot-che-2024", 168, 6150, "clips/pack-pl-tot-che-2024.json"),
    ("lei-mun-2015", 70, 5760, "clips/pack-3754186.json"),
    ("lei-avl-2015", 15, 6100, "clips/pack-3754106.json"),
    ("clasico-2017", 282, 6145, "clips/pack-267569.json"),
    ("bar-mal-2019", 243, 6000, "clips/pack-303451.json"),
    ("argfra-dimaria", 60, 9250, "clips/pack-argfra-2022.json"),
)

#: The seventh file. It is the control, not club football (study section 1).
FINAL = "argfra-dimaria"


def live(name: str, start: float, end: float) -> list[tuple[float, str]]:
    """Every utterance of one file's live window, brackets stripped."""
    found: list[tuple[float, str]] = []
    for utterance in utterances(load_json3(Path(f"clips/{name}.en.json3")).segments):
        if not start <= utterance.ts <= end:
            continue
        text = " ".join(_BRACKET.sub(" ", utterance.text).split())
        if text:
            found.append((utterance.ts, text))
    return found


def share[T](items: Sequence[T], predicate: Callable[[T], bool]) -> float:
    return sum(1 for item in items if predicate(item)) / len(items) if items else 0.0


def opener_repeats(texts: Sequence[str]) -> float:
    """The same count :func:`commentary.grading.register.opener_repeat_share` makes."""
    openers = [fold(text).split()[0] if fold(text).split() else "" for text in texts]
    repeats = sum(
        1
        for i, word in enumerate(openers)
        if word and word in openers[max(0, i - OPENER_WINDOW) : i]
    )
    return repeats / len(openers) if openers else 0.0


def names_of(path: str) -> Any:
    pack = KnowledgePack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    return name_words(frozenset(p.name for sheet in (pack.home, pack.away) for p in sheet.squad))


def report(label: str, files: Iterable[tuple[str, float, float, str]]) -> None:
    texts: list[str] = []
    gaps: list[float] = []
    bare = named = 0
    for name, start, end, pack_path in files:
        rows = live(name, start, end)
        words = names_of(pack_path)
        bare += sum(1 for _, text in rows if is_bare_name(text, words))
        named += sum(1 for _, text in rows if says_a_name(text, words))
        texts += [text for _, text in rows]
        gaps += [b - a for (a, _), (b, _) in zip(rows, rows[1:], strict=False)]
    counts = [len(text.split()) for text in texts]
    print(f"\n== {label}: {len(texts)} utterances, {sum(counts)} words")
    print(f"   median words          {statistics.median(counts):.0f}")
    print(f"   mean words            {statistics.mean(counts):.1f}")
    print(f"   one word              {share(counts, lambda n: n == 1):.3f}")
    print(f"   two or fewer          {share(counts, lambda n: n <= 2):.3f}   <- reference")
    print(f"   four or fewer         {share(counts, lambda n: n <= 4):.3f}")
    print(f"   nine or more          {share(counts, lambda n: n >= 9):.3f}")
    print(f"   sixteen or more       {share(counts, lambda n: n >= 16):.3f}")
    print(f"   median gap            {statistics.median(gaps):.2f}s")
    print(f"   gaps over 4s          {share(gaps, lambda g: g > 4):.3f}")
    print(f"   gaps over 8s          {share(gaps, lambda g: g > 8):.3f}   <- reference")
    print(f"   carries a number      {share(texts, lambda t: bool(NUMBER.search(t))):.3f}")
    off_score = share(texts, lambda t: bool(NUMBER.search(SCORELINE_SAID.sub(' ', t))))
    print(f"   a number, not a score {off_score:.3f}   <- reference")
    print(f"   opener repeated       {opener_repeats(texts):.3f}   <- reference")
    print(f"   bare name only        {bare / len(texts):.3f}")
    print(f"   carries a name        {named / len(texts):.3f}")


def main() -> int:
    club = [row for row in CORPUS if row[0] != FINAL]
    for row in CORPUS:
        report(row[0], [row])
    report("POOLED CLUB, the reference", club)
    report("POOLED CLUB without the radio feed", [r for r in club if r[0] != "lei-mun-2015"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
