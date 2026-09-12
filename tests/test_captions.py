"""Reading yt-dlp's caption file, which is the human commentary for free."""

from __future__ import annotations

from pathlib import Path

import pytest

from commentary.grading import captions, transcripts

FIXTURE = Path(__file__).parent / "fixtures" / "captions.en.json3"


@pytest.fixture
def loaded() -> transcripts.Transcript:
    return captions.load_json3(FIXTURE)


def test_every_event_with_text_becomes_one_segment(loaded: transcripts.Transcript):
    assert [s.text for s in loaded.segments] == [
        "and it's Di Maria on the left",
        "he's away",
        "GOAL",
    ]


def test_the_pieces_of_an_event_join_into_one_line(loaded: transcripts.Transcript):
    assert loaded.segments[0].start == pytest.approx(1.2)
    assert loaded.segments[0].end == pytest.approx(3.6)


def test_events_with_no_text_of_their_own_are_dropped(loaded: transcripts.Transcript):
    # The window definition, the bare newline and the empty seg are not speech,
    # and kept as segments they would read as three more lines the human said.
    assert len(loaded) == 3


def test_the_transcript_says_where_it_came_from(loaded: transcripts.Transcript):
    assert loaded.model == "youtube-auto"
    assert loaded.source == str(FIXTURE)
    assert loaded.duration_s == pytest.approx(11.0)


def test_it_saves_and_reloads_in_the_shape_the_eval_reads(
    loaded: transcripts.Transcript, tmp_path: Path
):
    out = tmp_path / "transcript.json"
    transcripts.save(loaded, out)
    again = transcripts.load(out)
    assert [(s.start, s.end, s.text) for s in again.segments] == [
        (s.start, s.end, s.text) for s in loaded.segments
    ]
    assert again.model == "youtube-auto"
