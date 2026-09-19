"""Offline contracts for the bounded streaming proof."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Frame, Observation, ObservedActor
from commentary.recorded_research import ResearchBrief, ResearchFact
from commentary.streaming_demo import (
    StreamingCommentator,
    StreamingLine,
    StreamingObserverResult,
    WriterResult,
)


def frames() -> list[Frame]:
    return [Frame(at_s=1.0, jpeg=b"one"), Frame(at_s=2.0, jpeg=b"two")]


def research() -> ResearchBrief:
    return ResearchBrief(
        fixture="Green v Orange",
        as_of=date(2026, 1, 1),
        facts=[
            ResearchFact(
                id="r1",
                subject="Green",
                text="Green had won two of three league matches.",
                source_url="https://example.com/green",
                source_date=date(2025, 12, 31),
            )
        ],
    )


@pytest.mark.asyncio
async def test_batch_resolves_identity_and_returns_auditable_exact_line() -> None:
    backend = ScriptedBackend()
    backend.always(
        "streaming_observer",
        StreamingObserverResult(
            observations=[
                Observation(
                    id="o1",
                    at_s=2,
                    kind="play",
                    description="A pass goes forward.",
                    actor=ObservedActor(
                        side="away",
                        shirt_number=7,
                        identity_source="shirt_number",
                        action_role="passer",
                    ),
                )
            ]
        ),
    )

    def writer(blocks: list[dict[str, object]], _format: type[object]) -> WriterResult:
        payload = json.loads(str(blocks[0]["text"]))
        assert payload["desired_voice"] == "caller"
        assert payload["observations"][0]["actor"]["resolved_name"] == "Ada Away"
        assert "roster" not in str(payload)
        return WriterResult(
            line=StreamingLine(voice="caller", text="Ada Away finds space.", evidence_ids=["b1-o1"])
        )

    backend.register("streaming_writer", writer)
    commentator = StreamingCommentator(
        backend,
        pack={"away": {"name": "Orange", "starters": [{"name": "Ada Away", "number": 7}]}},
        research=research(),
    )

    result = await commentator.process(frames(), playback_s=2)

    assert result["line"] == {
        "voice": "caller",
        "text": "Ada Away finds space.",
        "evidence_ids": ["b1-o1"],
        "research_ids": [],
    }
    assert result["observations"][0]["actor"]["resolved_name"] == "Ada Away"
    assert result["usage"]["call_delta"]["observer"]["output_tokens"] == 0
    assert result["audit"]["writer_raw"] == {"line": result["line"]}
    assert [call.tag for call in backend.calls] == ["streaming_observer", "streaming_writer"]


@pytest.mark.asyncio
async def test_goal_prioritizes_caller_and_rejects_unsupported_writer_evidence() -> None:
    backend = ScriptedBackend()
    backend.always(
        "streaming_observer",
        StreamingObserverResult(
            observations=[
                Observation(id="goal", at_s=2, kind="goal", description="The ball is in.")
            ]
        ),
    )
    backend.always(
        "streaming_writer",
        WriterResult(
            line=StreamingLine(voice="caller", text="Goal!", evidence_ids=["not-in-this-batch"])
        ),
    )

    with pytest.raises(ValueError, match="unsupported or future evidence"):
        await StreamingCommentator(backend).process(frames(), playback_s=2)


@pytest.mark.asyncio
async def test_empty_observation_batch_skips_writer_and_records_silence() -> None:
    backend = ScriptedBackend()
    backend.always("streaming_observer", StreamingObserverResult(observations=[]))

    result = await StreamingCommentator(backend).process(frames(), playback_s=2)

    assert result["line"] is None
    assert result["warnings"] == ["observer returned no observations; writer skipped"]
    assert [call.tag for call in backend.calls] == ["streaming_observer"]
    assert result["usage"]["call_delta"]["writer"]["cost_usd"] == 0


@pytest.mark.asyncio
async def test_empty_actor_object_is_dropped_without_losing_visual_observation() -> None:
    backend = ScriptedBackend()
    backend.always(
        "streaming_observer",
        StreamingObserverResult.model_validate(
            {
                "observations": [
                    {
                        "id": "unreadable",
                        "at_s": 2,
                        "kind": "play",
                        "description": "The ball moves forward.",
                        "actor": {
                            "side": "home",
                            "shirt_number": None,
                            "shirt_name": None,
                            "identity_source": "shirt_number",
                            "action_role": "passer",
                        },
                    }
                ]
            }
        ),
    )
    backend.always("streaming_writer", WriterResult(line=None))

    result = await StreamingCommentator(backend).process(frames(), playback_s=2)

    assert result["observations"][0]["actor"] is None
    assert result["audit"]["observer_actor_normalization"] == {
        "caveat": "empty actor objects are normalized to null before identity validation"
    }


def test_claimed_identity_still_uses_the_strict_recorded_actor_validation() -> None:
    with pytest.raises(ValidationError, match="shirt_name source requires shirt_name"):
        StreamingObserverResult.model_validate(
            {
                "observations": [
                    {
                        "id": "bad",
                        "at_s": 1,
                        "kind": "play",
                        "description": "A pass.",
                        "actor": {
                            "side": "home",
                            "shirt_number": 7,
                            "shirt_name": None,
                            "identity_source": "shirt_name",
                            "action_role": "passer",
                        },
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_history_rotates_to_analyst_after_two_to_one_caller_lead() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "streaming_observer",
        [
            StreamingObserverResult(
                observations=[Observation(id="o1", at_s=1, kind="play", description="Play.")]
            ),
            StreamingObserverResult(
                observations=[Observation(id="o2", at_s=2, kind="play", description="Play.")]
            ),
        ],
    )
    backend.queue(
        "streaming_writer",
        [
            WriterResult(
                line=StreamingLine(voice="caller", text="The move begins.", evidence_ids=["b1-o1"])
            ),
            WriterResult(
                line=StreamingLine(
                    voice="analyst", text="Space opens early.", evidence_ids=["b2-o1"]
                )
            ),
        ],
    )
    commentator = StreamingCommentator(backend)
    await commentator.process([Frame(at_s=1, jpeg=b"one")], playback_s=1)
    result = await commentator.process([Frame(at_s=2, jpeg=b"two")], playback_s=2)

    assert result["audit"]["desired_voice"] == "analyst"
    writer_payload = json.loads(backend.calls_tagged("streaming_writer")[1].text)
    assert len(writer_payload["recent_lines"]) == 1
    assert len(writer_payload["observations"]) == 2


@pytest.mark.asyncio
async def test_cheap_streaming_defaults_and_explicit_astra_reasoning() -> None:
    class CapturingBackend(ScriptedBackend):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[dict[str, object]] = []

        async def parse(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            return await super().parse(**kwargs)  # type: ignore[arg-type]

    async def process_once(backend: CapturingBackend, **models: str) -> None:
        backend.always(
            "streaming_observer",
            StreamingObserverResult(
                observations=[Observation(id="o", at_s=1, kind="play", description="Play.")]
            ),
        )
        backend.always(
            "streaming_writer",
            WriterResult(
                line=StreamingLine(voice="caller", text="The move starts.", evidence_ids=["b1-o1"])
            ),
        )
        await StreamingCommentator(backend, **models).process(
            [Frame(at_s=1, jpeg=b"one")], playback_s=1
        )

    defaults = CapturingBackend()
    await process_once(defaults)
    assert [
        (request["model"], request["effort"], request["max_tokens"])
        for request in defaults.requests
    ] == [
        ("gpt-5.6-terra", "none", 2048),
        ("gpt-5.6-terra", "none", 1024),
    ]

    astra_observer = CapturingBackend()
    await process_once(astra_observer, observer_model="gpt-6-astra")
    assert [request["effort"] for request in astra_observer.requests] == ["low", "none"]


@pytest.mark.asyncio
async def test_frames_must_be_bounded_and_strictly_ordered() -> None:
    commentator = StreamingCommentator(ScriptedBackend())
    with pytest.raises(ValueError, match="1 to 8"):
        await commentator.process([], playback_s=1)
    with pytest.raises(ValueError, match="strictly ordered"):
        await commentator.process(
            [Frame(at_s=1, jpeg=b"one"), Frame(at_s=1, jpeg=b"two")], playback_s=1
        )
    with pytest.raises(ValueError, match="available"):
        await commentator.process([Frame(at_s=2, jpeg=b"two")], playback_s=1)
