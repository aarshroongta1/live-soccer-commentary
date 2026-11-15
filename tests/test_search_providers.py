from datetime import date

import httpx
import pytest

from commentary.llm.fake import ScriptedBackend
from commentary.replay.preparation import prepare_research
from commentary.search_providers import (
    CitedSearchFact,
    SearchHit,
    SearchResearchDraft,
    SearchRun,
    search_web,
)


async def test_same_query_and_count_normalize_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "private-key")
    monkeypatch.setenv("EXA_API_KEY", "private-key")
    requests = []

    def answer(request):
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Source",
                        "url": "https://example.com/story",
                        "snippet": "Same excerpt",
                        "highlights": ["Same excerpt"],
                        "published_at": "2025-11-08",
                        "publishedDate": "2025-11-08",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        keen = await search_web("keenable", "shared query", client=client)
        exa = await search_web("exa", "shared query", client=client)
    assert len(requests) == 2
    assert requests[0]["query"] == requests[1]["query"] == "shared query"
    assert requests[0]["max_results"] == requests[1]["numResults"] == 5
    assert keen.results == exa.results
    assert "private-key" not in keen.model_dump_json() + exa.model_dump_json()


async def test_provider_failure_does_not_retry_or_fall_back(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "private-key")
    calls = []

    def answer(request):
        calls.append(request)
        return httpx.Response(429, text="private-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        with pytest.raises(RuntimeError, match="^exa search returned HTTP 429$"):
            await search_web("exa", "query", client=client)
    assert len(calls) == 1


def saved_search():
    return SearchRun(
        provider="exa",
        query="query",
        elapsed_ms=100,
        retrieved_at="2026-11-15",
        request={},
        results=[
            SearchHit(
                rank=1,
                title="Later",
                url="https://example.com/later",
                excerpt="Future information",
                published_at="2025-11-10",
            ),
            SearchHit(
                rank=2,
                title="Contemporary",
                url="https://example.com/source",
                excerpt="Ballard came through Arsenal's academy.",
                published_at="2025-11-08T10:00:00Z",
            ),
            SearchHit(rank=3, title="Undated", url="https://example.com/undated", excerpt="X"),
        ],
    )


async def test_research_excludes_later_and_undated_content_and_binds_source():
    backend = ScriptedBackend()
    backend.always(
        "replay_researcher",
        SearchResearchDraft(
            facts=[
                CitedSearchFact(
                    source_rank=2,
                    subject="Ballard",
                    text="Ballard came through Arsenal's academy.",
                    quote="Ballard came through Arsenal's academy.",
                )
            ]
        ),
    )
    prepared = await prepare_research(
        backend,
        fixture="Sunderland v Arsenal",
        as_of=date(2025, 11, 8),
        model="test",
        search_run=saved_search(),
    )
    assert "Future information" not in backend.calls[0].text
    assert "https://example.com/undated" not in backend.calls[0].text
    assert prepared.brief.facts[0].source_url == "https://example.com/source"
    assert prepared.brief.facts[0].source_date == date(2025, 11, 8)
    assert prepared.audit["source"] == "exa"


@pytest.mark.parametrize("rank,quote", [(1, "Future information"), (2, "Invented quote")])
async def test_research_rejects_invented_or_unavailable_citations(rank, quote):
    backend = ScriptedBackend()
    backend.always(
        "replay_researcher",
        SearchResearchDraft(
            facts=[
                CitedSearchFact(
                    source_rank=rank,
                    subject="Ballard",
                    text="A claim",
                    quote=quote,
                )
            ]
        ),
    )
    prepared = await prepare_research(
        backend,
        fixture="Sunderland v Arsenal",
        as_of=date(2025, 11, 8),
        model="test",
        search_run=saved_search(),
    )
    assert prepared.brief.facts == []
    assert prepared.audit["omitted_facts"] == [{"source_rank": rank, "reason": "unverified_quote"}]
