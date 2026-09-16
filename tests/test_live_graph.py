"""The lead graph routes typed results without owning live I/O."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from commentary.orchestration.context import (
    CallerResult,
    LeadGraphContext,
    ObservationResult,
    PhraseResult,
    VerificationResult,
)
from commentary.orchestration.live import build_live_commentary_graph
from commentary.orchestration.state import CommentaryTurnState, new_turn_state
from commentary.schemas import Beat, CallerLine, Event, GateVerdict, MatchState, Scene, Side, Voice


def form(*, speak: bool = True) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.CARRY,
        side=Side.HOME,
        confidence=0.9,
        speak=speak,
        line="Hale carries through midfield." if speak else "",
    )


@dataclass
class Services:
    caller: CallerResult = field(default_factory=lambda: CallerResult(form()))
    phrase: PhraseResult = field(
        default_factory=lambda: PhraseResult(
            form().model_copy(update={"line": "Hale, through midfield."}), 0.3
        )
    )
    passed: bool = True
    fact_version: int = 1
    observation: ObservationResult = field(default_factory=ObservationResult)
    calls: list[str] = field(default_factory=list)

    async def call_caller(self, state: CommentaryTurnState) -> CallerResult:
        self.calls.append("caller")
        return self.caller

    def observe_form(
        self, state: CommentaryTurnState, line: CallerLine
    ) -> ObservationResult:
        self.calls.append("observe")
        return self.observation

    async def phrase_candidate(
        self, state: CommentaryTurnState, line: CallerLine
    ) -> PhraseResult:
        self.calls.append("phrase")
        self.fact_version += 1
        return self.phrase

    def verify_candidate(
        self,
        state: CommentaryTurnState,
        original: CallerLine,
        candidate: CallerLine,
    ) -> VerificationResult:
        self.calls.append("verify")
        return VerificationResult(
            fact_version=self.fact_version,
            verdict=GateVerdict(passed=self.passed, line=candidate.line),
        )

    def build_beat(
        self,
        state: CommentaryTurnState,
        original: CallerLine,
        candidate: CallerLine,
        verdict: GateVerdict,
        excitement: float,
    ) -> Beat:
        self.calls.append("build")
        return Beat(
            id=f"lead:{state['turn_id']}",
            voice=Voice.CALLER,
            text=verdict.line,
            video_ts=state["cursor_s"],
            live_ts=state["live_s"],
            created_ts=0.0,
            event=original.event,
            excitement=excitement,
            triggers=[],
        )


def turn() -> CommentaryTurnState:
    return new_turn_state(
        match_id="match-1",
        turn_id="turn-1",
        cursor_s=10.0,
        live_s=18.0,
        triggers=["scheduled"],
        fact_version=1,
        fact_summary="Home 0-0 Away",
        match_state=MatchState(home="Home", away="Away"),
        goal_in_state=False,
    )


@pytest.mark.asyncio
async def test_the_happy_path_returns_a_serializable_beat() -> None:
    services = Services()
    graph = build_live_commentary_graph()

    result = await graph.ainvoke(turn(), context=LeadGraphContext(services))

    assert result["outcome"] == "ready"
    assert result["beat"]["id"] == "lead:turn-1"
    assert result["beat"]["text"] == "Hale, through midfield."
    assert services.calls == ["caller", "observe", "phrase", "verify", "build"]


@pytest.mark.asyncio
async def test_a_caller_silence_still_updates_observations() -> None:
    services = Services(caller=CallerResult(form(speak=False)))
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["outcome"] == "silent"
    assert result["beat"] is None
    assert services.calls == ["caller", "observe"]


@pytest.mark.asyncio
async def test_a_caller_failure_stops_before_observation() -> None:
    services = Services(caller=CallerResult(None, "provider failed"))
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["outcome"] == "failed"
    assert result["error"] == "provider failed"
    assert services.calls == ["caller"]


@pytest.mark.asyncio
async def test_an_observation_can_stop_the_turn() -> None:
    services = Services(observation=ObservationResult(False, "replay spent"))
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["outcome"] == "silent"
    assert result["error"] == "replay spent"
    assert services.calls == ["caller", "observe"]


@pytest.mark.asyncio
async def test_a_phraser_silence_skips_verification() -> None:
    services = Services(phrase=PhraseResult(None, silent=True, error="chosen silence"))
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["outcome"] == "silent"
    assert result["error"] == "chosen silence"
    assert services.calls == ["caller", "observe", "phrase"]


@pytest.mark.asyncio
async def test_a_refusal_never_builds_a_beat() -> None:
    services = Services(passed=False)
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["outcome"] == "refused"
    assert result["beat"] is None
    assert services.calls == ["caller", "observe", "phrase", "verify"]


@pytest.mark.asyncio
async def test_verification_records_facts_refreshed_after_the_model_call() -> None:
    services = Services(fact_version=7)
    result = await build_live_commentary_graph().ainvoke(
        turn(), context=LeadGraphContext(services)
    )

    assert result["input_fact_version"] == 1
    assert result["verified_fact_version"] == 8
