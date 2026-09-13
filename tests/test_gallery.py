"""The gallery, on fakes: no weights, no network, no pictures of anybody.

What is under test is the rule, not SigLIP. A fake embedder answers with a
fixed vector per "player", so the question each test asks is whether the
threshold and the margin let a name out at the right times — which is the
whole of what makes this safe to run.
"""

import numpy as np
import pytest

from commentary.perception.gallery import MIN_BODY_PX, Gallery
from commentary.schemas import Side

#: Three players, as vectors far enough apart to be told apart, plus one that
#: sits between two of them the way two players of a build in one kit do.
VECTORS = {
    "messi": (1.0, 0.0, 0.0),
    "dimaria": (0.0, 1.0, 0.0),
    "mbappe": (0.0, 0.0, 1.0),
}


class FakeEmbedder:
    """A vector per crop, looked up by the tag painted into its first pixel."""

    def __init__(self) -> None:
        self.calls = 0
        self.crops = 0

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self.calls += 1
        self.crops += len(crops)
        return np.asarray([crop[0, 0].astype(np.float64) for crop in crops], dtype=np.float32)


def crop(vector: tuple[float, float, float], height: int = MIN_BODY_PX) -> np.ndarray:
    """A crop carrying its own embedding, so the fake needs no bookkeeping."""
    image = np.zeros((height, 20, 3), dtype=np.float64)
    image[:, :] = vector
    return image


def between(a: str, b: str, lean: float = 0.5) -> np.ndarray:
    first, second = np.asarray(VECTORS[a]), np.asarray(VECTORS[b])
    return crop(tuple(first * lean + second * (1.0 - lean)))


def gallery(**kwargs: float) -> Gallery:
    return Gallery(FakeEmbedder(), **kwargs)  # type: ignore[arg-type]


def taught(**kwargs: float) -> Gallery:
    g = gallery(**kwargs)
    g.learn(crop(VECTORS["messi"]), Side.HOME, 10, "Lionel Messi")
    g.learn(crop(VECTORS["dimaria"]), Side.HOME, 11, "Ángel Di María")
    return g


def test_two_confirmed_sightings_build_a_gallery_of_two():
    assert taught().players == 2


def test_a_body_near_one_player_and_far_from_the_other_is_named():
    found = taught().classify([(crop(VECTORS["dimaria"]), Side.HOME)])
    assert found[0] is not None
    assert (found[0].number, found[0].name) == (11, "Ángel Di María")


def test_a_body_between_two_players_is_not_named():
    """The margin, which is the whole safety argument.

    Two players of a build in the same kit sit between each other's
    centroids. The answer there is silence, not a coin toss with a name on
    it: the failure that matters is not missing a name, it is calling
    somebody by a team-mate's.
    """
    found = taught().classify([(between("messi", "dimaria"), Side.HOME)])
    assert found[0] is None


def test_a_body_that_looks_like_nobody_is_not_named():
    found = taught().classify([(crop(VECTORS["mbappe"]), Side.HOME)])
    assert found[0] is None


def test_a_body_is_only_matched_against_its_own_team():
    g = gallery()
    g.learn(crop(VECTORS["messi"]), Side.HOME, 10, "Lionel Messi")
    found = g.classify([(crop(VECTORS["messi"]), Side.AWAY)])
    assert found[0] is None, "an Argentina body cannot be a France player"


def test_a_body_on_neither_team_is_matched_against_everybody():
    """The kit split fails on close-ups, which is where recognition matters."""
    found = taught().classify([(crop(VECTORS["messi"]), Side.UNKNOWN)])
    assert found[0] is not None
    assert found[0].name == "Lionel Messi"


def test_a_body_too_small_to_recognise_is_never_embedded():
    g = taught()
    before = g.embedder.calls  # type: ignore[attr-defined]
    found = g.classify([(crop(VECTORS["messi"], height=MIN_BODY_PX - 1), Side.HOME)])
    assert found == [None]
    assert g.embedder.calls == before, "a thirty-pixel body has nothing in it"  # type: ignore[attr-defined]


def test_an_empty_gallery_names_nobody():
    assert gallery().classify([(crop(VECTORS["messi"]), Side.HOME)]) == [None]


def test_only_a_read_sighting_teaches_it():
    """There is no other way in, by construction: nothing else calls learn."""
    g = taught()
    g.classify([(crop(VECTORS["messi"]), Side.HOME)])
    assert g.players == 2


def test_the_last_eight_samples_are_what_is_remembered():
    g = gallery()
    for _ in range(12):
        g.learn(crop(VECTORS["messi"]), Side.HOME, 10, "Lionel Messi")
    assert g.players == 1


@pytest.mark.parametrize("lean", [0.55, 0.6])
def test_a_looser_margin_is_what_lets_a_close_call_through(lean: float):
    """Stated as a test so the constants are not quietly tuned into a guess."""
    tight = taught(margin=0.5)
    loose = taught(margin=0.01)
    body = between("dimaria", "messi", lean)
    assert tight.classify([(body, Side.HOME)])[0] is None
    assert loose.classify([(body, Side.HOME)])[0] is not None


def test_the_trace_row_carries_every_decision_and_its_margin():
    g = taught()
    g.classify([(crop(VECTORS["dimaria"]), Side.HOME), (crop(VECTORS["mbappe"]), Side.HOME)])
    row = g.drain()
    assert row["players"] == 2
    assert row["classified"] == 2
    assert row["hits"] == 1
    assert row["named"] == ["Ángel Di María"]
    assert len(row["margins"]) == 2
    assert g.drain()["classified"] == 0, "drained rows are not reported twice"
