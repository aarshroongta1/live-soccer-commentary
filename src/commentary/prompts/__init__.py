"""What the models are actually shown.

The prompts live apart from the agents because they change on a different
clock. A prompt here is a deliverable in its own right — the caller's rules
about lookahead, names, the scoreline and silence are most of what separates
this system from a frame narrator — and keeping them in one module means a
prompt sweep is a diff of one file, and the eval can rebuild exactly the
strings the runtime sent.
"""

from commentary.prompts.caller import caller_blocks, caller_system

__all__ = ["caller_blocks", "caller_system"]
