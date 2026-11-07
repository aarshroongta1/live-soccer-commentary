"""Official split identity, offline provenance, and manifest integrity checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from importlib.resources import files
from pathlib import Path

import pytest

from commentary.replay.soccernet_splits import (
    _parse_index,
    canonical_match_id,
    official_splits,
    verify_split,
)

SAMPLE = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"


def test_official_snapshot_contains_every_partition_and_verified_sample() -> None:
    index = official_splits()
    assert Counter(index.assignments.values()) == {
        "train": 300,
        "valid": 100,
        "test": 100,
        "challenge": 50,
    }
    assert verify_split(SAMPLE + "/", "train") == "train"
    assert index.revision == "95d76e2b46b43738dbf18f2d7ab3d7283d5aeff5"
    raw = files("commentary.replay").joinpath("data/soccernet-events-v2-splits.json").read_bytes()
    assert index.sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(TypeError):
        index.assignments[SAMPLE] = "test"  # type: ignore[index]


def test_challenge_matches_cannot_be_reassigned_as_an_ordinary_split() -> None:
    index = official_splits()
    match = next(match for match, split in index.assignments.items() if split == "challenge")
    with pytest.raises(ValueError, match="challenge matches cannot"):
        verify_split(match, "test")


def test_matching_uses_full_identity_and_handles_windows_slashes() -> None:
    assert canonical_match_id(SAMPLE.replace("/", "\\") + "\\") == SAMPLE
    with pytest.raises(ValueError, match="unknown in the pinned"):
        verify_split(SAMPLE.replace("england_epl", "other_league"), "train")
    with pytest.raises(ValueError, match="competition/season/match"):
        verify_split(SAMPLE.rsplit("/", 1)[-1], "train")


@pytest.mark.parametrize("corruption", ["duplicate", "overlap", "missing_split", "empty_split"])
def test_corrupt_manifest_never_silently_verifies_an_assignment(corruption: str) -> None:
    path = files("commentary.replay").joinpath("data/soccernet-events-v2-splits.json")
    data = json.loads(path.read_text())
    if corruption == "duplicate":
        data["splits"]["train"].append(data["splits"]["train"][0])
    elif corruption == "overlap":
        data["splits"]["test"].append(data["splits"]["train"][0])
    elif corruption == "missing_split":
        del data["splits"]["challenge"]
    else:
        data["splits"]["valid"] = []
    with pytest.raises(ValueError):
        _parse_index(json.dumps(data).encode())


def test_wrong_split_fails_before_reading_local_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts/import_soccernet.py"
    spec = importlib.util.spec_from_file_location("import_soccernet_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = tmp_path / "Labels-v2.json"
    labels.write_text(json.dumps({"UrlLocal": SAMPLE, "annotations": []}))
    output = tmp_path / "case.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_soccernet",
            "--labels",
            str(labels),
            "--half",
            "1",
            "--upstream-split",
            "valid",
            "--clip",
            str(tmp_path / "not-downloaded.mkv"),
            "--start",
            "0",
            "--duration",
            "10",
            "--video-half-offset",
            "0",
            "--case-id",
            "sample",
            "--label-origin",
            "synthetic://test-labels",
            "--usage-terms",
            "Test fixture.",
            "--out",
            str(output),
        ],
    )
    monkeypatch.setattr(
        module, "validate_inputs", lambda *_args: pytest.fail("video read too soon")
    )
    with pytest.raises(ValueError, match="official split 'train'"):
        module.main()
    assert not output.exists()
