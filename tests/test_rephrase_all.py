"""Discovery and mapping logic in ``scripts/rephrase_all.py``. Free, no spend.

The script lives under ``scripts/``, not ``src/``, so it is loaded here from
its file path rather than imported as a package — the same reason
``scripts/voice_sweep.py`` has no test of its own, except that this module's
discovery and mapping logic is pure and worth pinning down: a wrong pack
mapping is a wrong roster silently trimming every surname, and it is the
kind of mistake a spend-gated script would otherwise only surface after
paying for it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "rephrase_all.py"
_spec = importlib.util.spec_from_file_location("rephrase_all", _SCRIPT)
assert _spec is not None and _spec.loader is not None
rephrase_all = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rephrase_all
_spec.loader.exec_module(rephrase_all)


def test_run_key_of_nested_directory() -> None:
    trace = rephrase_all.RUNS_ROOT / "sonnet" / "e01-counter" / "file-20260913-185319.jsonl"
    assert rephrase_all.run_key_of(trace) == "sonnet/e01-counter"


def test_run_key_of_flat_directory() -> None:
    trace = rephrase_all.RUNS_ROOT / "penalty1" / "file-20260912-231231.jsonl"
    assert rephrase_all.run_key_of(trace) == "penalty1"


def test_run_key_of_no_subdirectory_strips_timestamp() -> None:
    trace = rephrase_all.RUNS_ROOT / "sim-20260911-185124.jsonl"
    assert rephrase_all.run_key_of(trace) == "sim"


@pytest.mark.parametrize(
    ("run_key", "model"),
    [
        ("e09-offside", "opus"),
        ("mbappe", "opus"),
        ("sonnet", "sonnet"),
        ("sonnet/e01-counter", "sonnet"),
        ("abl/marks-e01-counter-r1", "haiku"),
        ("screen/dimaria", "haiku"),
        ("voice/dimaria-goal", "haiku"),
        # `voice/mbappe-v2` is a *different* trace under the same top-level
        # directory as the one Haiku override, and is not itself overridden:
        # the longest-prefix rule must not let the directory-level "screen"
        # or "voice" entries swallow siblings that were never named.
        ("voice/mbappe-v2", "opus"),
    ],
)
def test_model_of(run_key: str, model: str) -> None:
    assert rephrase_all.model_of(run_key) == model


def test_every_opus_pack_mapping_points_at_a_real_pack_file() -> None:
    """Every mapped pack path exists on disk, from the worktree root.

    Catches the one mistake this table is entirely capable of making by
    hand: a pack file renamed or moved after the table was written.
    """
    for run_key, pack in rephrase_all.CLIP_TO_PACK.items():
        assert Path(pack).exists(), f"{run_key} -> {pack} does not exist"


def test_clip_to_pack_has_no_sonnet_or_haiku_only_keys() -> None:
    """The pack table is keyed by run-key regardless of model; `model_of` is
    the only thing that excludes Sonnet and Haiku traces. A run-key present
    in both tables is fine — `discover` checks the model first — so this
    only pins down that lookup order, not that the tables are disjoint.
    """
    sonnet_keys = {"sonnet", "sonnet/e01-counter", "sonnet/e02-penalty-scored"}
    for key in sonnet_keys:
        assert rephrase_all.model_of(key) == "sonnet"
