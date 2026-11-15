"""Interchangeable, single-request retrieval for the existing research agent.

Keep request bodies, ranked excerpts and raw responses for an honest replay.
No retries, fallback provider, API keys in artifacts, or provider-side synthesis.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

ProviderName = Literal["keenable", "exa"]


class SearchHit(BaseModel):
    rank: int
    title: str
    url: str
    excerpt: str
    published_at: str | None = None


class SearchRun(BaseModel):
    provider: ProviderName
    query: str
    retrieved_at: str
    elapsed_ms: float
    request: dict[str, Any]
    results: list[SearchHit]
    raw: dict[str, Any] = Field(default_factory=dict)


class CitedSearchFact(BaseModel):
    source_rank: int = Field(ge=1, le=10)
    subject: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=360)
    quote: str = Field(min_length=1, max_length=1400)


class SearchResearchDraft(BaseModel):
    facts: list[CitedSearchFact] = Field(max_length=6)


async def search_web(
    provider: ProviderName,
    query: str,
    *,
    max_results: int = 5,
    client: httpx.AsyncClient | None = None,
) -> SearchRun:
    """Equal query and result count; each provider's general-purpose default mode.

    Five results are a deliberate small demo budget. Both agents receive at most
    2,500 characters per hit. Full original responses remain in the run artifact.
    """
    if provider not in {"keenable", "exa"}:
        raise ValueError("unknown search provider")
    if not query.strip() or not 1 <= max_results <= 10:
        raise ValueError("search needs a nonblank query and 1–10 results")
    key = os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not key:
        raise ValueError(f"{provider.upper()}_API_KEY is missing")
    if provider == "keenable":
        url = "https://api.keenable.ai/v1/search"
        body: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "snippet_max_length": 2500,
        }
    else:
        url = "https://api.exa.ai/search"
        body = {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "objective": "Find source-backed football background for a recorded commentary "
            "agent. Prefer contemporary reporting and club sources; extract relevant player "
            "history and team form, excluding unrelated fixtures.",
            "contents": {"highlights": True},
        }
    started = time.monotonic()

    async def request(http: httpx.AsyncClient) -> dict[str, Any]:
        response = await http.post(url, headers={"x-api-key": key}, json=body)
        if not response.is_success:
            # Avoid echoing arbitrary response text or authenticated request headers.
            raise RuntimeError(f"{provider} search returned HTTP {response.status_code}")
        data: dict[str, Any] = response.json()
        return data

    if client is None:
        async with httpx.AsyncClient(timeout=30) as http:
            raw = await request(http)
    else:
        raw = await request(client)
    elapsed = (time.monotonic() - started) * 1000
    hits = []
    for rank, item in enumerate(raw.get("results", [])[:max_results], 1):
        excerpt = (
            item.get("snippet", "")
            if provider == "keenable"
            else "\n".join(item.get("highlights") or [])
        )
        hits.append(
            SearchHit(
                rank=rank,
                title=item.get("title") or "Untitled",
                url=item["url"],
                excerpt=excerpt[:2500],
                published_at=item.get("published_at") or item.get("publishedDate"),
            )
        )
    return SearchRun(
        provider=provider,
        query=query,
        retrieved_at=datetime.now(UTC).isoformat(),
        elapsed_ms=round(elapsed, 1),
        request=body,
        results=hits,
        raw=raw,
    )
