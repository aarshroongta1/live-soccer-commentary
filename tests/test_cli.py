"""The command line's own logic: the bits that are not just argparse wiring."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from commentary.__main__ import _crop_preview, build_parser, crop_box


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
