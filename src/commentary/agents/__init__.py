"""The voices.

An agent here is three things: the frames it is entitled to see, the prompt it
is given, and the rules it must obey after the model has spoken. The third is
the part that keeps a live match honest — a model that rambles, repeats itself
or narrates a replay is a model doing its job badly, and the agent is where
that gets caught rather than shipped to a voice.
"""

from commentary.agents.caller import Caller, RepetitionGate, clean_line, similarity, trim_words

__all__ = ["Caller", "RepetitionGate", "clean_line", "similarity", "trim_words"]
