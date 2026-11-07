"""Pinned official event-spotting partitions; reference import only, no network."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OfficialSplit = Literal["train", "valid", "test", "challenge"]


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["SoccerNet events-v2"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source: str
    sources: dict[OfficialSplit, _Source]
    splits: dict[OfficialSplit, list[str]]


def canonical_match_id(value: str) -> str:
    """Keep the full dataset identity; a match basename alone is insufficient."""
    match = value.replace("\\", "/").rstrip("/")
    parts = match.split("/")
    if len(parts) != 3 or any(not part.strip() or part in {".", ".."} for part in parts):
        raise ValueError("match identity must contain competition/season/match")
    return match


@dataclass(frozen=True)
class SplitIndex:
    assignments: Mapping[str, OfficialSplit]
    source: str
    revision: str
    sha256: str


def _parse_index(raw: bytes) -> SplitIndex:
    manifest = _Manifest.model_validate_json(raw)
    expected = {"train", "valid", "test", "challenge"}
    if set(manifest.splits) != expected or set(manifest.sources) != expected:
        raise ValueError("official split manifest must contain all four partitions")
    assignments: dict[str, OfficialSplit] = {}
    for split, matches in manifest.splits.items():
        if not matches:
            raise ValueError("official split partitions must not be empty")
        for value in matches:
            match = canonical_match_id(value)
            if match in assignments:
                raise ValueError(f"duplicate match in official split manifest: {match}")
            assignments[match] = split
    return SplitIndex(
        MappingProxyType(assignments),
        manifest.source,
        manifest.revision,
        hashlib.sha256(raw).hexdigest(),
    )


@lru_cache(maxsize=1)
def official_splits() -> SplitIndex:
    raw = files("commentary.replay").joinpath("data/soccernet-events-v2-splits.json").read_bytes()
    return _parse_index(raw)


def verify_split(match_id: str, claimed: str) -> OfficialSplit:
    official = official_splits().assignments.get(canonical_match_id(match_id))
    if official is None:
        raise ValueError(
            "match is unknown in the pinned SoccerNet events-v2 split manifest; "
            "verify the dataset/release and update the official manifest before importing"
        )
    if official == "challenge":
        raise ValueError("challenge matches cannot be imported as development, validation, or test")
    if claimed != official:
        raise ValueError(f"claimed split {claimed!r} disagrees with official split {official!r}")
    return official
