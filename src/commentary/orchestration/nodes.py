"""Thin LangGraph nodes for one lead-commentary turn."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.runtime import Runtime as GraphRuntime

from commentary.orchestration.context import LeadGraphContext
from commentary.orchestration.state import CommentaryTurnState
from commentary.schemas import CallerLine, GateVerdict


async def call_caller(
    state: CommentaryTurnState,
    runtime: GraphRuntime[LeadGraphContext],
) -> dict[str, Any]:
    result = await runtime.context.services.call_caller(state)
    if result.form is None:
        return {"outcome": "failed", "error": result.error or "caller returned nothing"}
    return {"caller_form": result.form.model_dump(mode="json")}


def after_caller(state: CommentaryTurnState) -> Literal["observe", "done"]:
    return "observe" if state["caller_form"] is not None else "done"


def observe_form(
    state: CommentaryTurnState,
    runtime: GraphRuntime[LeadGraphContext],
) -> dict[str, Any]:
    form = CallerLine.model_validate(state["caller_form"])
    result = runtime.context.services.observe_form(state, form)
    updates: dict[str, Any] = {}
    if result.form is not None:
        updates["caller_form"] = result.form.model_dump(mode="json")
    if not result.continue_turn or not form.speak or not form.line.strip():
        updates.update({"outcome": "silent", "error": result.error})
    return updates


def after_observe(state: CommentaryTurnState) -> Literal["phrase", "done"]:
    return "phrase" if state["outcome"] == "pending" else "done"


async def phrase_candidate(
    state: CommentaryTurnState,
    runtime: GraphRuntime[LeadGraphContext],
) -> dict[str, Any]:
    form = CallerLine.model_validate(state["caller_form"])
    result = await runtime.context.services.phrase_candidate(state, form)
    if result.silent:
        return {"outcome": "silent", "error": result.error}
    if result.candidate is None:
        return {"outcome": "failed", "error": result.error or "phrasing produced nothing"}
    return {
        "candidate": result.candidate.model_dump(mode="json"),
        "excitement": result.excitement,
        "error": result.error,
    }


def after_phrase(state: CommentaryTurnState) -> Literal["verify", "done"]:
    return "verify" if state["candidate"] is not None else "done"


def verify_candidate(
    state: CommentaryTurnState,
    runtime: GraphRuntime[LeadGraphContext],
) -> dict[str, Any]:
    form = CallerLine.model_validate(state["caller_form"])
    candidate = CallerLine.model_validate(state["candidate"])
    result = runtime.context.services.verify_candidate(state, form, candidate)
    outcome = "pending" if result.verdict.passed else "refused"
    return {
        "verified_fact_version": result.fact_version,
        "verdict": result.verdict.model_dump(mode="json"),
        "outcome": outcome,
    }


def after_verify(state: CommentaryTurnState) -> Literal["build", "done"]:
    verdict = state["verdict"]
    return "build" if verdict is not None and GateVerdict.model_validate(verdict).passed else "done"


def build_beat(
    state: CommentaryTurnState,
    runtime: GraphRuntime[LeadGraphContext],
) -> dict[str, Any]:
    form = CallerLine.model_validate(state["caller_form"])
    candidate = CallerLine.model_validate(state["candidate"])
    verdict = GateVerdict.model_validate(state["verdict"])
    beat = runtime.context.services.build_beat(
        state,
        form,
        candidate,
        verdict,
        state["excitement"],
    )
    return {"beat": beat.model_dump(mode="json"), "outcome": "ready"}
