"""The voices.

An agent here is three things: the frames it is entitled to see, the prompt it
is given, and the rules it must obey after the model has spoken. The third is
the part that keeps a live match honest — a model that rambles, repeats itself
or narrates a replay is a model doing its job badly, and the agent is where
that gets caught rather than shipped to a voice.

The researcher is the odd one out. It is the only agent allowed to reach the
outside world, and it is an exception in time rather than in kind: it runs
once before kickoff, writes a knowledge pack, and that pack is frozen when the
whistle goes.
"""

from commentary.agents.analyst import Analyst
from commentary.agents.caller import Caller, RepetitionGate, clean_line, similarity, trim_words
from commentary.agents.researcher import (
    Researcher,
    freeze,
    is_frozen,
    load_pack,
    pack_path,
    save_pack,
    update_from_substitution,
)

__all__ = [
    "Analyst",
    "Caller",
    "RepetitionGate",
    "Researcher",
    "clean_line",
    "freeze",
    "is_frozen",
    "load_pack",
    "pack_path",
    "save_pack",
    "similarity",
    "trim_words",
    "update_from_substitution",
]
