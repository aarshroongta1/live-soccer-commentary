import asyncio

from commentary.live_research import CommentaryText, PassageChoice, research_events
from commentary.llm.fake import ScriptedBackend
from commentary.search_providers import SearchHit, SearchRun


def result(provider):
    return SearchRun(
        provider=provider,
        query="same query",
        retrieved_at="2026-11-15",
        elapsed_ms=5 if provider == "keenable" else 10,
        request={},
        results=[
            SearchHit(
                rank=1,
                title="Source",
                url="https://example.com/story",
                excerpt="Arsenal had eight clean sheets in all competitions before this match.",
                published_at="2025-11-08",
            )
        ],
    )


async def test_other_search_can_return_while_first_fact_is_being_processed():
    backend = ScriptedBackend(latency_s=0.03)
    backend.always("live_fact_selector", PassageChoice(passage_id="r1-p0"))
    backend.always(
        "live_commentary",
        CommentaryText(text="Eight clean sheets in all competitions, and that run is over."),
    )
    requests = []

    async def search(provider, query):
        requests.append(query)
        await asyncio.sleep(0.005 if provider == "keenable" else 0.01)
        return result(provider)

    async def voice(text):
        return {"audio_base64": "test", "text": text}

    events = [event async for event in research_events(backend, search=search, synthesize=voice)]
    returns = [event for event in events if event["type"] == "search_returned"]
    assert [event["provider"] for event in returns] == ["keenable", "exa"]
    assert len(set(requests)) == 1
    assert events.index(returns[1]) < next(
        i for i, e in enumerate(events) if e["type"] == "fact_selected"
    )
    fact = next(e["fact"] for e in events if e["type"] == "fact_selected")
    writer = backend.calls_tagged("live_commentary")[0]
    assert fact["text"] in writer.text
    assert fact["url"] in writer.text
    assert len(backend.calls) == 2
    line = next(e["text"] for e in events if e["type"] == "commentary_ready")
    spoken = next(e["text"] for e in events if e["type"] == "audio_ready")
    assert spoken == line  # No editorial rewrite between display and synthesis.


async def test_completion_order_is_not_forced_and_invalid_fact_cannot_reach_writer():
    backend = ScriptedBackend()
    backend.always("live_fact_selector", PassageChoice(passage_id="invented"))

    async def search(provider, query):
        await asyncio.sleep(0.01 if provider == "keenable" else 0.001)
        return result(provider)

    async def voice(text):
        raise AssertionError("No voice call is allowed for ungrounded output")

    events = [e async for e in research_events(backend, search=search, synthesize=voice)]
    assert next(e for e in events if e["type"] == "search_returned")["provider"] == "exa"
    assert next(e for e in events if e["type"] == "pipeline_error")
    assert not backend.calls_tagged("live_commentary")


async def test_workflow_or_provider_narration_is_never_synthesized():
    backend = ScriptedBackend()
    backend.always("live_fact_selector", PassageChoice(passage_id="r1-p0"))
    backend.always(
        "live_commentary", CommentaryText(text="Keenable returned this fact faster than Exa.")
    )

    async def search(provider, query):
        return result(provider)

    async def voice(text):
        raise AssertionError("Promotional narration must not reach voice synthesis")

    events = [e async for e in research_events(backend, search=search, synthesize=voice)]
    assert any(e["type"] == "pipeline_error" for e in events)
    assert not any(e["type"] == "audio_ready" for e in events)
