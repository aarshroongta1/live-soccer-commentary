"""Injected services for the bounded lead-commentary graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from commentary.orchestration.state import CommentaryTurnState
from commentary.schemas import Beat, CallerLine, GateVerdict


@dataclass(frozen=True)
class CallerResult:
    form: CallerLine | None
    error: str = ""


@dataclass(frozen=True)
class PhraseResult:
    candidate: CallerLine | None
    excitement: float = 0.0
    silent: bool = False
    error: str = ""


@dataclass(frozen=True)
class ObservationResult:
    continue_turn: bool = True
    error: str = ""


@dataclass(frozen=True)
class VerificationResult:
    fact_version: int
    verdict: GateVerdict


class LeadTurnServices(Protocol):
    """The graph's narrow port into models and deterministic domain code."""

    async def call_caller(self, state: CommentaryTurnState) -> CallerResult: ...

    def observe_form(
        self, state: CommentaryTurnState, form: CallerLine
    ) -> ObservationResult: ...

    async def phrase_candidate(
        self, state: CommentaryTurnState, form: CallerLine
    ) -> PhraseResult: ...

    def verify_candidate(
        self,
        state: CommentaryTurnState,
        form: CallerLine,
        candidate: CallerLine,
    ) -> VerificationResult: ...

    def build_beat(
        self,
        state: CommentaryTurnState,
        form: CallerLine,
        candidate: CallerLine,
        verdict: GateVerdict,
        excitement: float,
    ) -> Beat: ...


@dataclass(frozen=True)
class LeadGraphContext:
    services: LeadTurnServices
