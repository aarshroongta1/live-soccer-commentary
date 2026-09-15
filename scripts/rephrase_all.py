"""Rephrase every Opus caller trace on disk, and pool the register report.

One trace against a rewrite is eight points of noise (see the module this
script exists to fix: `docs/HANDOFF.md` section on the register judge). The
honest measurement of a phraser change is not one trace re-read by hand, it
is the pooled set of every real Opus caller trace `runs/` holds. This script
finds those traces, maps each to the knowledge pack its run used, runs
`commentary rephrase` over the ones that are not already done, and then pools
the free-layer register numbers — median words, bare-name share, the gaps,
the per-event table — over every phrased line at once rather than averaging
per-trace shares, which is not the same arithmetic once trace lengths differ.

    uv run python scripts/rephrase_all.py --dry-run
    uv run python scripts/rephrase_all.py --tag v6
    uv run python scripts/rephrase_all.py --tag v6 --limit 2 --force
    uv run python scripts/rephrase_all.py --tag v6 --compare v5

Traces land in `runs/rephrased/<tag>/<run-key>/`, mirroring the discovered
trace's own path under `runs/` so two tags never collide and a rerun with
`--force` overwrites cleanly. `POOLED.md` and `POOLED.json` sit at
`runs/rephrased/<tag>/`; the JSON is what `--compare` reads back.

Discovery defaults to Opus caller traces only, because the phraser is judged
on Opus forms (`HANDOFF.md`, "Sonnet versus Opus... Stay on Opus for the
caller"). Sonnet and Haiku traces are recognised and always reported, but
only `--include-haiku` will phrase the Haiku ones too; Sonnet traces are
never included here (a separate one-off comparison, not the corpus).

Interruptible: Ctrl-C between two `rephrase` subprocess calls leaves every
finished output in place, and the pooling step at the end reads whatever
exists on disk, not what this run intended.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.gate import fold
from commentary.grading import register as reg
from commentary.grading.register import Refusal, RegisterReport, Shape, Source, Utterance
from commentary.trace import read_trace, rows_of

RUNS_ROOT = Path("runs")
REPHRASED_ROOT = RUNS_ROOT / "rephrased"
CLIPS_ROOT = Path("clips")

#: What one phrased line costs, roughly — the v3 measurement in HANDOFF.md
#: section 3d is $0.0012 a line on a cached prefix; $0.0015 leaves headroom
#: for the calls that miss the cache.
PHRASE_LINE_COST_USD = 0.0015

#: How far back an opener is compared for repetition, mirrored from
#: `commentary.grading.register` so pooling can recompute the same flag per
#: trace without reaching into a private name.
OPENER_WINDOW = reg.OPENER_WINDOW

#: run-key -> knowledge pack, read off `docs/CLIPS.md` and `docs/HANDOFF.md`
#: by hand. A run-key is the discovered trace's parent directory under
#: `runs/`, or for a trace with no parent subdirectory (the sim runs) the
#: filename stem with its timestamp stripped. Nothing here is inferred at
#: runtime because the pack a run used is not recorded in the trace itself —
#: checked: no topic row on any trace carries a model or a pack name.
ARGFRA = "clips/pack-argfra-2022.json"  # Argentina v France, WC22 final
NED_ARG = "clips/pack-3869321.json"  # Netherlands v Argentina, WC22 QF
MOR_ESP = "clips/pack-3869220.json"  # Morocco v Spain, WC22 R16
POR_ESP = "clips/pack-7576.json"  # Portugal v Spain, WC18
CRO_ENG = "clips/pack-8656.json"  # Croatia v England, WC18 SF
ARG_COL = "clips/pack-3943077.json"  # Argentina v Colombia, Copa 2024 final

CLIP_TO_PACK: dict[str, str] = {
    # The tuned clip (Argentina v France 2022, Di María's goal) and its many
    # reruns, ablations and calibration passes.
    "A": ARGFRA,
    "B": ARGFRA,
    "c11": ARGFRA,
    "c12": ARGFRA,
    "c12b": ARGFRA,
    "cadence/mbappe": ARGFRA,
    "card": ARGFRA,
    "marks": ARGFRA,
    "nomarks": ARGFRA,
    "mbappe": ARGFRA,
    "offside": ARGFRA,
    "opus": ARGFRA,
    "penalty1": ARGFRA,
    "r2": ARGFRA,
    "r3": ARGFRA,
    "r4": ARGFRA,
    "r5": ARGFRA,
    "screen/dimaria": ARGFRA,
    "shootout": ARGFRA,
    "sonnet": ARGFRA,
    "subs": ARGFRA,
    "trigger/mbappe": ARGFRA,
    "voice/dimaria-goal": ARGFRA,
    # NED-ARG WC22: e01 counter, e02 penalty, e03 shootout winner, e04
    # penalty save.
    "e01-counter": NED_ARG,
    "e01b-counter": NED_ARG,
    "e02-penalty-scored": NED_ARG,
    "e02b-penalty": NED_ARG,
    "e03-shootout-winner": NED_ARG,
    "e04-penalty-save": NED_ARG,
    "sonnet/e01-counter": NED_ARG,
    "sonnet/e02-penalty-scored": NED_ARG,
    "abl/marks-e01-counter-r1": NED_ARG,
    "abl/marks-e01-counter-r2": NED_ARG,
    "abl/nomarks-e01-counter-r1": NED_ARG,
    "abl/nomarks-e01-counter-r2": NED_ARG,
    "abl/marks-e02-penalty-scored-r1": NED_ARG,
    "abl/marks-e02-penalty-scored-r2": NED_ARG,
    "abl/nomarks-e02-penalty-scored-r1": NED_ARG,
    "abl/nomarks-e02-penalty-scored-r2": NED_ARG,
    # MOR-ESP WC22: e05 shootout miss, e06 yellow card.
    "e05-shootout-save": MOR_ESP,
    "e06-yellow": MOR_ESP,
    # POR-ESP WC18: e07 free-kick goal, e08 penalty.
    "e07-freekick": POR_ESP,
    "e07b-freekick": POR_ESP,
    "e08-penalty-early": POR_ESP,
    "sonnet/e07-freekick": POR_ESP,
    "sonnet/e08-penalty-early": POR_ESP,
    "abl/marks-e07-freekick-r1": POR_ESP,
    "abl/marks-e07-freekick-r2": POR_ESP,
    "abl/nomarks-e07-freekick-r1": POR_ESP,
    "abl/nomarks-e07-freekick-r2": POR_ESP,
    "abl/marks-e08-penalty-early-r1": POR_ESP,
    "abl/marks-e08-penalty-early-r2": POR_ESP,
    "abl/nomarks-e08-penalty-early-r1": POR_ESP,
    "abl/nomarks-e08-penalty-early-r2": POR_ESP,
    # CRO-ENG WC18: e09 offside, e11 corner.
    "e09-offside": CRO_ENG,
    "e11-corner": CRO_ENG,
    "sonnet/e09-offside": CRO_ENG,
    "prompt-name/e09-offside": CRO_ENG,
    "prompt-name/e11-corner": CRO_ENG,
    "w1280/e09-offside": CRO_ENG,
    "w1280/e11-corner": CRO_ENG,
    "abl/marks-e09-offside-r1": CRO_ENG,
    "abl/marks-e09-offside-r2": CRO_ENG,
    "abl/nomarks-e09-offside-r1": CRO_ENG,
    "abl/nomarks-e09-offside-r2": CRO_ENG,
    "abl/marks-e11-corner-r1": CRO_ENG,
    "abl/marks-e11-corner-r2": CRO_ENG,
    "abl/nomarks-e11-corner-r1": CRO_ENG,
    "abl/nomarks-e11-corner-r2": CRO_ENG,
    # ARG-COL Copa 2024 final: e10 offside, e12 foul.
    "e10-offside-copa": ARG_COL,
    "e12-foul": ARG_COL,
}

#: run-key prefixes whose caller was not Opus. Read off HANDOFF.md by hand,
#: the same way `CLIP_TO_PACK` was: section 3c names the screen run and the
#: dimaria-goal replay "Haiku caller" outright; section 7 names the tracker
#: ablation "Haiku in both arms"; `runs/sonnet/` is named for what it is.
#: Longest prefix wins, so a specific path (`voice/dimaria-goal`) beats its
#: directory (`voice`, which also holds `replay-` traces the path filter
#: already drops).
MODEL_PREFIX: dict[str, str] = {
    "sonnet": "sonnet",
    "abl": "haiku",
    "screen": "haiku",
    "voice/dimaria-goal": "haiku",
}

_TIMESTAMP_SUFFIX = re.compile(r"-\d{8}-\d{6}$")


def run_key_of(trace: Path) -> str:
    """The discovered trace's identity for pack lookup, output layout and dedup.

    The parent directory under ``runs/``, or — for the handful of traces
    sitting directly in ``runs/`` with no subdirectory of their own, which is
    only the simulator traces — the filename with its timestamp trimmed off.
    """
    rel = trace.relative_to(RUNS_ROOT)
    if len(rel.parts) > 1:
        return rel.parent.as_posix()
    return _TIMESTAMP_SUFFIX.sub("", trace.stem)


def model_of(run_key: str) -> str:
    best = ""
    for prefix in MODEL_PREFIX:
        if (run_key == prefix or run_key.startswith(prefix + "/")) and len(prefix) > len(best):
            best = prefix
    return MODEL_PREFIX.get(best, "opus")


# -- discovery ------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    run_key: str
    trace: Path
    pack: str
    model: str
    speak_lines: int


@dataclass
class Discovery:
    included: list[Candidate] = field(default_factory=list)
    excluded_model: list[Candidate] = field(default_factory=list)
    unmapped: list[tuple[str, Path, str]] = field(default_factory=list)  # key, trace, model


def discover(*, include_haiku: bool) -> Discovery:
    """Every original caller trace under ``runs/``, sorted by lines ascending.

    "Original" means the filename is stamped ``file-``, ``screen-`` or
    ``sim-`` — the three sources ``commentary run`` writes
    (``runtime.trace_path``) — and not ``replay-``, which is what
    ``commentary replay`` writes over a trace that was already phrased or
    voiced. ``runs/rephrased/`` is this script's own output tree and is
    never walked.
    """
    out = Discovery()
    for trace in sorted(RUNS_ROOT.rglob("*.jsonl")):
        if REPHRASED_ROOT in trace.parents:
            continue
        if trace.name.startswith("replay-"):
            continue
        rows = read_trace(trace)
        caller_rows = rows_of(rows, "caller")
        speak_lines = sum(1 for row in caller_rows if row.get("speak", True))
        if speak_lines == 0:
            continue
        run_key = run_key_of(trace)
        model = model_of(run_key)
        pack = CLIP_TO_PACK.get(run_key)
        if model == "sonnet" or (model == "haiku" and not include_haiku):
            out.excluded_model.append(Candidate(run_key, trace, pack or "", model, speak_lines))
            continue
        if pack is None:
            out.unmapped.append((run_key, trace, model))
            continue
        out.included.append(Candidate(run_key, trace, pack, model, speak_lines))
    out.included.sort(key=lambda c: c.speak_lines)
    return out


# -- dry run ----------------------------------------------------------------


def print_discovery(disc: Discovery, *, limit: int | None) -> None:
    shown = disc.included if limit is None else disc.included[:limit]
    print(f"{len(disc.included)} Opus candidate(s) mapped to a pack "
          f"({len(shown)} selected by this invocation):")
    for c in shown:
        est = c.speak_lines * PHRASE_LINE_COST_USD
        print(f"  {c.run_key:<28} {str(c.trace):<48} {c.pack:<32} "
              f"{c.speak_lines:3d} lines  ~${est:.4f}")
    total_lines = sum(c.speak_lines for c in shown)
    total_est = total_lines * PHRASE_LINE_COST_USD
    print(f"total: {total_lines} lines, ~${total_est:.4f} at ${PHRASE_LINE_COST_USD}/line")

    if disc.excluded_model:
        by_model: dict[str, list[Candidate]] = {}
        for c in disc.excluded_model:
            by_model.setdefault(c.model, []).append(c)
        print()
        for model, items in sorted(by_model.items()):
            flag = "pass --include-haiku to add" if model == "haiku" else "never included here"
            print(f"excluded, {model} caller ({flag}): "
                  + ", ".join(c.run_key for c in items))

    if disc.unmapped:
        print()
        print("could not map to a pack, skipped:")
        for run_key, trace, model in disc.unmapped:
            print(f"  {run_key:<28} {trace} (model guess: {model})")


# -- running ------------------------------------------------------------


def output_path(tag: str, candidate: Candidate) -> Path:
    out_dir = REPHRASED_ROOT / tag / candidate.run_key
    return out_dir / f"{candidate.trace.stem}-phrased.jsonl"


def run_rephrase(candidate: Candidate, tag: str, *, colour: bool, force: bool) -> bool:
    """Run ``commentary rephrase`` for one candidate. True if output exists after."""
    out_path = output_path(tag, candidate)
    if out_path.exists() and not force:
        print(f"== {candidate.run_key}: already done ({out_path}), skipping")
        return True
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== {candidate.run_key} <- {candidate.trace}  ({candidate.speak_lines} lines)")
    cmd = [
        "uv", "run", "python", "-m", "commentary", "rephrase",
        "--trace", str(candidate.trace),
        "--pack", candidate.pack,
        "--out", str(out_dir),
        "--colour" if colour else "--no-colour",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"!! {candidate.run_key}: rephrase exited {result.returncode}")
    return out_path.exists()


def run_batch(
    candidates: list[Candidate], tag: str, *, colour: bool, force: bool
) -> list[Candidate]:
    """Run every candidate in order; Ctrl-C leaves finished ones in place.

    Returns the candidates whose output actually exists on disk afterwards —
    whatever ran this call plus whatever was already there — which is what
    gets pooled next.
    """
    done: list[Candidate] = []
    try:
        for candidate in candidates:
            if run_rephrase(candidate, tag, colour=colour, force=force):
                done.append(candidate)
    except KeyboardInterrupt:
        print(f"\ninterrupted: {len(done)} of {len(candidates)} candidates have output on disk")
    return done


# -- pooling ----------------------------------------------------------------


@dataclass(frozen=True)
class LineFact:
    utt: Utterance
    bare: bool
    name: bool
    number: bool
    number_off_score: bool
    opener_repeat: bool


def _share(n: int, total: int) -> float:
    return n / total if total else 0.0


def trace_facts(
    path: Path, pack: Any
) -> tuple[list[LineFact], list[Utterance], list[Refusal], int, float]:
    """Everything `pooled_shape` needs from one phrased trace, per line.

    Bare-name and name-share checks need the trace's own team sheet — two
    matches never share a roster — so each line is tagged here, against its
    own trace's names, before anything is pooled across traces.
    """
    rows = read_trace(path)
    spoken, _source = reg.lines_of(rows)
    lead = [u for u in spoken if u.seat == "lead"]
    colour = [u for u in spoken if u.seat != "lead"]
    people = reg.people_of(rows, pack)
    names = reg.name_words(people)
    refused = reg.refusals_of(rows)

    openers = [fold(u.text).split()[0] if fold(u.text).split() else "" for u in lead]
    facts: list[LineFact] = []
    for i, u in enumerate(lead):
        opener = openers[i]
        repeat = bool(opener) and opener in openers[max(0, i - OPENER_WINDOW) : i]
        facts.append(
            LineFact(
                utt=u,
                bare=reg.is_bare_name(u.text, names),
                name=reg.says_a_name(u.text, names),
                number=bool(reg.NUMBER.search(u.text)),
                number_off_score=reg.says_a_number_off_score(u.text),
                opener_repeat=repeat,
            )
        )
    duration = max((u.ts for u in spoken), default=0.0)
    return facts, colour, refused, len(people), duration


def pooled_by_event(facts: list[LineFact]) -> list[reg.EventShape]:
    grouped: dict[str, list[LineFact]] = {}
    for f in facts:
        grouped.setdefault(f.utt.event, []).append(f)
    shapes = [
        reg.EventShape(
            event=event,
            lines=len(group),
            median_words=float(statistics.median([g.utt.words for g in group])),
            share_le_4=_share(sum(1 for g in group if g.utt.words <= 4), len(group)),
            bare_name_share=_share(sum(1 for g in group if g.bare), len(group)),
            name_share=_share(sum(1 for g in group if g.name), len(group)),
        )
        for event, group in grouped.items()
    ]
    return sorted(shapes, key=lambda s: (-s.lines, s.event))


def phrased_usd(path: Path) -> float:
    """What the phrasing itself cost. The `phrased` rows' `usd` field, never `cost`."""
    rows = read_trace(path)
    return sum(float(row.get("usd", 0.0)) for row in rows if row.get("topic") == "phrased")


@dataclass
class Pooled:
    shape: Shape
    per_trace: list[tuple[str, Path, Shape, float]]  # run_key, path, shape, usd
    total_usd: float


def pool(done: list[Candidate], tag: str, packs: dict[str, Any]) -> Pooled:
    all_facts: list[LineFact] = []
    all_colour: list[Utterance] = []
    all_refused: list[Refusal] = []
    known_people_total = 0
    duration_total = 0.0
    gaps: list[float] = []
    per_trace: list[tuple[str, Path, Shape, float]] = []
    total_usd = 0.0

    for candidate in done:
        out_path = output_path(tag, candidate)
        pack = packs[candidate.pack]
        rows = read_trace(out_path)
        shape_i = reg.measure(rows, pack)
        usd_i = phrased_usd(out_path)
        per_trace.append((candidate.run_key, out_path, shape_i, usd_i))
        total_usd += usd_i

        facts, colour, refused, known, duration = trace_facts(out_path, pack)
        all_facts.extend(facts)
        all_colour.extend(colour)
        all_refused.extend(refused)
        known_people_total += known
        duration_total += duration

        spoken_i = sorted([f.utt for f in facts] + colour, key=lambda u: u.ts)
        gaps.extend(b.ts - a.ts for a, b in zip(spoken_i, spoken_i[1:], strict=False))

    lead_utts = [f.utt for f in all_facts]
    pooled = Shape(
        source=Source.PHRASED,
        lines=len(lead_utts),
        colour_lines=len(all_colour),
        lead=lead_utts,
        colour=all_colour,
        refused=all_refused,
        known_people=known_people_total,
        duration_s=duration_total,
    )
    for r in all_refused:
        pooled.refusal_reasons[r.tag] = pooled.refusal_reasons.get(r.tag, 0) + 1
    judged = len(lead_utts) + len(all_refused)
    pooled.gate_refused_share = _share(len(all_refused), judged)

    if lead_utts:
        counts = [u.words for u in lead_utts]
        pooled.median_words = float(statistics.median(counts))
        pooled.share_le_2 = _share(sum(1 for n in counts if n <= 2), len(counts))
        pooled.share_le_4 = _share(sum(1 for n in counts if n <= 4), len(counts))
        pooled.share_ge_9 = _share(sum(1 for n in counts if n >= 9), len(counts))
        pooled.share_ge_16 = _share(sum(1 for n in counts if n >= 16), len(counts))
        pooled.bare_name_share = _share(sum(1 for f in all_facts if f.bare), len(all_facts))
        pooled.name_share = _share(sum(1 for f in all_facts if f.name), len(all_facts))
        pooled.opener_repeat_share = _share(
            sum(1 for f in all_facts if f.opener_repeat), len(all_facts)
        )
        pooled.number_share = _share(sum(1 for f in all_facts if f.number), len(all_facts))
        pooled.number_share_off_score = _share(
            sum(1 for f in all_facts if f.number_off_score), len(all_facts)
        )
        pooled.by_event = pooled_by_event(all_facts)

    if gaps:
        pooled.median_gap_s = float(statistics.median(gaps))
        pooled.share_gap_gt_4 = _share(sum(1 for g in gaps if g > 4.0), len(gaps))
        pooled.share_gap_gt_8 = _share(sum(1 for g in gaps if g > 8.0), len(gaps))

    return Pooled(shape=pooled, per_trace=per_trace, total_usd=total_usd)


def per_trace_table(pooled: Pooled) -> str:
    rows = [f"{'run':<28}{'lines':>7}{'colour':>8}{'words':>8}{'bare':>7}{'name':>7}{'usd':>9}"]
    rows.append("-" * 74)
    for run_key, _path, shape_i, usd_i in pooled.per_trace:
        rows.append(
            f"{run_key:<28}{shape_i.lines:>7}{shape_i.colour_lines:>8}"
            f"{shape_i.median_words:>8.1f}{shape_i.bare_name_share * 100:>6.0f}%"
            f"{shape_i.name_share * 100:>6.0f}%{usd_i:>9.4f}"
        )
    return "\n".join(rows)


async def write_pooled_report(
    pooled: Pooled, tag: str, *, judge: bool, timeout_s: float
) -> RegisterReport:
    n = len(pooled.per_trace)
    report = RegisterReport(name=f"POOLED tag={tag} ({n} traces)", shape=pooled.shape)

    if judge and pooled.shape.lead:
        from commentary.config import JUDGE_MODEL
        from commentary.llm import grading_backend

        backend = grading_backend(timeout_s, no_key_hint="the judge needs a key")
        verdict, usage = await reg.judge_register(
            pooled.shape, backend, name=f"pooled/{tag}", model=JUDGE_MODEL
        )
        report.verdict = verdict
        report.usage = usage
        report.model = JUDGE_MODEL

    out_dir = REPHRASED_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "POOLED.md"
    lines = [
        f"# Pooled register report, tag `{tag}`",
        "",
        f"{n} trace(s) pooled, {pooled.shape.lines} lead lines, "
        f"{pooled.shape.colour_lines} colour lines.",
        f"Phrasing spend (the `usd` field on `phrased` rows, not `cost` rows): "
        f"${pooled.total_usd:.4f}",
        "",
        "## Per-trace",
        "",
        "```",
        per_trace_table(pooled),
        "```",
        "",
        "## Pooled free layer and per-event breakdown",
        "",
        "```",
        report.table(),
        "```",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.write(out_dir / "POOLED.json")
    print(f"\nwrote {md_path}")
    return report


# -- compare ------------------------------------------------------------


def compare(tag: str, other: str) -> int:
    a_path = REPHRASED_ROOT / tag / "POOLED.json"
    b_path = REPHRASED_ROOT / other / "POOLED.json"
    if not a_path.exists() or not b_path.exists():
        missing = a_path if not a_path.exists() else b_path
        print(f"missing {missing}: run this script with --tag on it first (no --compare needed)")
        return 1
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    print(f"{tag} vs {other}")
    print(f"{'':<28}{tag:>10}{other:>10}{'delta':>10}")
    for key, band in a["measured"].items():
        va = band["trace"]
        vb = b["measured"].get(key, {}).get("trace")
        if vb is None:
            continue
        delta = va - vb
        unit = band.get("unit", "")

        def fmt(v: float, unit: str = unit) -> str:
            return f"{v * 100:.0f}%" if unit == "share" else f"{v:.1f}"

        print(f"{band['what']:<28}{fmt(va):>10}{fmt(vb):>10}{fmt(delta):>10}")
    a_judge, b_judge = a.get("judge"), b.get("judge")
    if isinstance(a_judge, dict) and isinstance(b_judge, dict):
        print()
        print(f"{'judge':<28}{tag:>10}{other:>10}{'delta':>10}")
        for label in a_judge.get("scores", {}):
            sa = a_judge["scores"].get(label, {}).get("score")
            sb = b_judge.get("scores", {}).get(label, {}).get("score")
            if sa is None or sb is None:
                continue
            print(f"{label:<28}{sa:>10.1f}{sb:>10.1f}{sa - sb:>+10.1f}")
    else:
        print("\n(no --judge run on one or both tags: judge scores not compared)")
    return 0


# -- main ------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    if args.compare:
        return compare(args.tag, args.compare)

    disc = discover(include_haiku=args.include_haiku)

    if args.dry_run:
        print_discovery(disc, limit=args.limit)
        return 0

    candidates = disc.included if args.limit is None else disc.included[: args.limit]
    print_discovery(disc, limit=args.limit)
    if not candidates:
        print("nothing to do")
        return 0

    done = run_batch(candidates, args.tag, colour=args.colour, force=args.force)
    if not done:
        print("nothing on disk to pool")
        return 1

    packs: dict[str, Any] = {}
    from commentary.agents.researcher import load_pack

    for pack_path in {c.pack for c in done}:
        packs[pack_path] = load_pack(Path(pack_path))

    pooled = pool(done, args.tag, packs)
    await write_pooled_report(pooled, args.tag, judge=args.judge, timeout_s=args.timeout)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", help="required unless --dry-run or --compare: runs/rephrased/<tag>/")
    p.add_argument("--dry-run", action="store_true", help="list candidates and the cost estimate")
    p.add_argument(
        "--limit", type=int, default=None, help="phrase only the first N (smallest) candidates"
    )
    p.add_argument(
        "--force", action="store_true", help="re-run even if this tag's output already exists"
    )
    p.add_argument(
        "--colour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="offer the colour seat (default on)",
    )
    p.add_argument(
        "--include-haiku", action="store_true", help="also phrase traces whose caller was Haiku"
    )
    p.add_argument(
        "--judge", action="store_true", help="also run the paid Opus judge on the pooled set"
    )
    p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the judge")
    p.add_argument(
        "--compare", help="print this tag's pooled table beside another tag's, with deltas"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.tag:
        print("--tag is required unless --dry-run or --compare", file=sys.stderr)
        return 2
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
