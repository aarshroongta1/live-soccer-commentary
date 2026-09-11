"""What the models are actually shown.

The prompts live apart from the agents because they change on a different
clock. A prompt here is a deliverable in its own right — the caller's rules
about lookahead, names, the scoreline and silence are most of what separates
this system from a frame narrator — and keeping them in one module means a
prompt sweep is a diff of one file, and the grader can rebuild exactly the
strings the runtime sent.

Every system prompt in here is byte-stable for a given knowledge pack. That
is a hard requirement rather than a nicety: the caller's prefix carries two
full team sheets and is sent every few seconds for ninety minutes, so it only
pays for itself if it caches, and it only caches if nothing in it drifts.
"""

from commentary.prompts.analyst import analyst_blocks, analyst_system
from commentary.prompts.caller import caller_blocks, caller_system
from commentary.prompts.researcher import researcher_blocks, researcher_system

__all__ = [
    "analyst_blocks",
    "analyst_system",
    "caller_blocks",
    "caller_system",
    "researcher_blocks",
    "researcher_system",
]
