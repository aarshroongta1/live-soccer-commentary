"""The human commentary, on disk and sliceable, with no model in sight."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from commentary.grading import transcripts
from commentary.grading.transcripts import Segment, Transcript, TranscriptionUnavailable
from commentary.schemas import KnowledgePack, Player, TeamSheet


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(0.0, 4.0, "Arsenal begin with the ball"),
            Segment(28.0, 33.5, "Saka down the right"),
            Segment(33.5, 40.0, "and it is in! Saka scores"),
            Segment(70.0, 74.0, "a corner for Madrid"),
        ],
        source="half1.wav",
    )


@pytest.fixture
def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(name="Arsenal", starters=[Player(name="Martín Ødegaard", number=8)]),
        away=TeamSheet(name="Real Madrid", starters=[Player(name="Jude Bellingham", number=5)]),
    )


def test_save_and_load_round_trip(tmp_path: Path, transcript: Transcript) -> None:
    path = tmp_path / "commentary.json"
    transcripts.save(transcript, path)
    back = transcripts.load(path)
    assert back.segments == transcript.segments
    assert back.source == "half1.wav"
    assert back.model == transcripts.DEFAULT_MODEL
    assert back.duration_s == pytest.approx(74.0)


def test_segments_near_returns_what_overlaps_the_window(transcript: Transcript) -> None:
    near = transcripts.segments_near(transcript, 34.0, window_s=2.0)
    assert [s.text for s in near] == ["Saka down the right", "and it is in! Saka scores"]


def test_a_sentence_that_straddles_the_moment_still_counts(transcript: Transcript) -> None:
    """A goal call starts before the goal and runs past it; containment misses it."""
    assert transcripts.segments_near(transcript, 36.0, window_s=0.5)[0].text.endswith("scores")


def test_a_quiet_moment_returns_nothing(transcript: Transcript) -> None:
    assert transcripts.segments_near(transcript, 55.0, window_s=5.0) == []
    assert transcripts.said_near(transcript, 55.0, window_s=5.0) == ""


def test_said_near_joins_the_window(transcript: Transcript) -> None:
    assert transcripts.said_near(transcript, 34.0, window_s=2.0) == (
        "Saka down the right and it is in! Saka scores"
    )


def test_the_roster_prompt_carries_teams_and_names(pack: KnowledgePack) -> None:
    prompt = transcripts.roster_prompt(pack)
    assert "Arsenal versus Real Madrid" in prompt
    assert "Martín Ødegaard" in prompt and "Jude Bellingham" in prompt
    assert len(prompt) <= transcripts.PROMPT_LIMIT


def test_a_bare_list_of_names_works_as_a_roster() -> None:
    assert transcripts.roster_prompt(["Saka", "Rice"]) == "Players: Saka, Rice."
    assert transcripts.roster_prompt(None) == ""


def test_without_mlx_whisper_you_get_a_sentence_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(TranscriptionUnavailable) as caught:
        transcripts.transcribe(tmp_path / "half1.wav")
    message = str(caught.value)
    assert "mlx-whisper is not installed" in message
    assert "uv sync --extra eval" in message
