"""The bounded LangGraph workflow for one lead-commentary opportunity."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from commentary.orchestration.context import LeadGraphContext
from commentary.orchestration.nodes import (
    after_caller,
    after_observe,
    after_phrase,
    after_verify,
    build_beat,
    call_caller,
    observe_form,
    phrase_candidate,
    verify_candidate,
)
from commentary.orchestration.state import CommentaryTurnState


def build_live_commentary_graph(*, checkpointer: Any = None) -> Any:
    """Compile the lead graph; callers own the checkpointer lifecycle."""
    graph = StateGraph(CommentaryTurnState, context_schema=LeadGraphContext)
    graph.add_node("call_caller", call_caller)
    graph.add_node("observe_form", observe_form)
    graph.add_node("phrase_candidate", phrase_candidate)
    graph.add_node("verify_candidate", verify_candidate)
    graph.add_node("build_beat", build_beat)

    graph.add_edge(START, "call_caller")
    graph.add_conditional_edges(
        "call_caller", after_caller, {"observe": "observe_form", "done": END}
    )
    graph.add_conditional_edges(
        "observe_form", after_observe, {"phrase": "phrase_candidate", "done": END}
    )
    graph.add_conditional_edges(
        "phrase_candidate", after_phrase, {"verify": "verify_candidate", "done": END}
    )
    graph.add_conditional_edges(
        "verify_candidate", after_verify, {"build": "build_beat", "done": END}
    )
    graph.add_edge("build_beat", END)
    return graph.compile(checkpointer=checkpointer, name="live_commentary")
