"""The command line's own logic: the bits that are not just argparse wiring."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pytest

from commentary.__main__ import _crop_preview, _pack_for_air, build_parser, cmd_notes, crop_box
from commentary.agents.researcher import load_pack, researched_path, save_pack
from commentary.config import JUDGE_MODEL
from commentary.schemas import Note


def test_a_crop_box_is_four_fractions():
    assert crop_box("0,0,0.42,0.16") == (0.0, 0.0, 0.42, 0.16)


@pytest.mark.parametrize("text", ["0,0,0.42", "a,b,c,d", "0.5,0,0.4,0.2", "0,0,1.2,0.2"])
def test_a_crop_box_that_is_not_a_box_is_refused(text: str):
    with pytest.raises(argparse.ArgumentTypeError):
        crop_box(text)


def test_run_takes_the_crop_off_the_command_line():
    args = build_parser().parse_args(["run", "--crop", "0.1,0.2,0.3,0.4"])
    assert args.crop == (0.1, 0.2, 0.3, 0.4)


def test_the_preview_puts_the_bug_beside_the_marked_frame():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[5:15, 10:60] = 200  # something bug-shaped inside the box
    preview = _crop_preview(image, (0.0, 0.0, 0.4, 0.2))
    assert preview.shape == (100, 300, 3)
    # The box is drawn on a copy, and the original frame is untouched.
    assert int(image[:, :, 2].max()) == 200
    # The blown-up bug fills the right-hand panel.
    assert int(preview[:, 200:].max()) > 0


def test_notes_reads_a_pack_and_knows_which_command_it_is():
    args = build_parser().parse_args(["notes", "--pack", "clips/pack-x.json"])
    assert args.pack == "clips/pack-x.json"
    assert args.out is None
    assert args.func is cmd_notes


def test_notes_can_write_somewhere_else():
    args = build_parser().parse_args(
        ["notes", "--pack", "clips/pack-x.json", "--out", "packs/with-notes.json"]
    )
    assert args.out == "packs/with-notes.json"


def test_notes_needs_a_pack_to_add_notes_to():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["notes"])


def test_register_defaults_to_the_judge_and_takes_more_than_one_trace():
    args = build_parser().parse_args(["register", "a.jsonl", "b.jsonl"])
    assert args.traces == ["a.jsonl", "b.jsonl"]
    assert args.no_model is False
    assert args.model == JUDGE_MODEL


def test_register_can_be_run_for_nothing():
    args = build_parser().parse_args(["register", "a.jsonl", "--no-model"])
    assert args.no_model is True


def test_notes_writes_beside_the_pack_rather_than_over_it():
    """The default destination is the researched copy, never the pack itself.

    A pack with hand-checked notes in it took somebody an evening; a notes
    pass takes a model ninety seconds. Making the model's output the default
    destination is how the evening gets lost.
    """
    args = build_parser().parse_args(["notes", "--pack", "clips/pack-x.json"])
    assert args.out is None
    assert researched_path(Path(args.pack)) == Path("clips/pack-x-researched.json")


def test_notes_refuses_to_write_over_a_pack_somebody_has_checked(tmp_path):
    pack = load_pack(Path("clips/pack-argfra-2022.json"))
    target = tmp_path / "pack.json"
    save_pack(pack, target)
    args = build_parser().parse_args(
        ["notes", "--pack", str(target), "--out", str(target)]
    )
    with pytest.raises(SystemExit, match="hand-checked"):
        asyncio.run(cmd_notes(args))


def test_run_and_rephrase_both_know_how_to_trust_an_unchecked_pack():
    assert build_parser().parse_args(["run"]).trust_unchecked is False
    args = build_parser().parse_args(["rephrase", "--trace", "t.jsonl", "--trust-unchecked"])
    assert args.trust_unchecked is True


def test_a_pack_for_air_leaves_the_unchecked_notes_behind(tmp_path, capsys):
    pack = load_pack(Path("clips/pack-argfra-2022.json"))
    notes = [*pack.notes, Note(about="Lionel Messi", text="nine goals this year")]
    target = tmp_path / "pack.json"
    save_pack(pack.model_copy(update={"notes": notes}), target)

    trimmed = _pack_for_air(target)
    assert len(trimmed.notes) == 13
    assert "1 of 14 notes are unchecked" in capsys.readouterr().out

    trusted = _pack_for_air(target, trust_unchecked=True)
    assert len(trusted.notes) == 14


def test_run_and_rephrase_both_take_a_trust_floor():
    from commentary.config import SETTINGS

    assert build_parser().parse_args(["run"]).trust_floor == SETTINGS.researcher.trust_floor
    args = build_parser().parse_args(
        ["rephrase", "--trace", "t.jsonl", "--trust-unchecked", "--trust-floor", "0.6"]
    )
    assert args.trust_floor == 0.6


def test_a_pack_for_air_trusts_only_what_clears_the_floor(tmp_path, capsys):
    """The real failure: a 0.1-confidence note, known wrong, aired at 20.5s."""
    pack = load_pack(Path("clips/pack-argfra-2022.json"))
    notes = [
        *pack.notes,
        Note(about="Lionel Messi", text="on a hat-trick tonight", confidence=0.1),
        Note(about="Lionel Messi", text="nine goals this year", confidence=0.9),
    ]
    target = tmp_path / "pack.json"
    save_pack(pack.model_copy(update={"notes": notes}), target)

    trusted = _pack_for_air(target, trust_unchecked=True, trust_floor=0.6)
    said = {note.text for note in trusted.notes}
    assert "on a hat-trick tonight" not in said
    assert "nine goals this year" in said
    out = capsys.readouterr().out
    assert "1 of 15 notes are unchecked and below the trust floor (0.6)" in out
