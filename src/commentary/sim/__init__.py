"""A broadcast of a match that never happened, with the answers in the back.

Nothing downstream of this package knows it is a simulation. The source hands
out the same ``Frame`` objects a screen capture does, the pictures carry a
score bug in the same crop a real one would, and the oracle is handed content
blocks like any other backend. What the sim adds is the thing a real
broadcast cannot give us: a written record of everything that happened, which
is what turns "the commentary sounded fine" into a number.
"""

from commentary.sim.audio import MatchAudio, band_energy
from commentary.sim.match import (
    PITCH_L,
    PITCH_W,
    Dot,
    MatchSim,
    Outcome,
    Phase,
    SimState,
)
from commentary.sim.oracle import (
    ERROR_KINDS,
    INJECTED_KINDS,
    OUTCOME_KIND,
    Moment,
    Sighting,
    SimOracle,
)
from commentary.sim.render import (
    TS_BLOCKS,
    BroadcastRenderer,
    decode_ts,
    encode_ts,
    kit_colour,
    ts_block_size,
)
from commentary.sim.source import SimSource
from commentary.sim.tracker import SimTracker

__all__ = [
    "ERROR_KINDS",
    "INJECTED_KINDS",
    "OUTCOME_KIND",
    "PITCH_L",
    "PITCH_W",
    "TS_BLOCKS",
    "BroadcastRenderer",
    "Dot",
    "MatchAudio",
    "MatchSim",
    "Moment",
    "Outcome",
    "Phase",
    "Sighting",
    "SimOracle",
    "SimSource",
    "SimState",
    "SimTracker",
    "band_energy",
    "decode_ts",
    "encode_ts",
    "kit_colour",
    "ts_block_size",
]
