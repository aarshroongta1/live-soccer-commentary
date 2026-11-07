"""Concurrency-safe reservations for bounded paid development runs.

Reservations include the maximum requested output, not just observed usage. A
cancelled/failed request retains its reservation because its billing is unknown.
This guards standard token spend, not the provider account's final invoice.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from commentary.llm.base import LLMError


class BudgetExceeded(LLMError):
    """No further paid request fits the configured development budget."""


class RunBudget:
    def __init__(self, *, max_usd: float = 0.10, max_calls: int = 40) -> None:
        if not math.isfinite(max_usd) or max_usd <= 0 or max_calls < 1:
            raise ValueError("budget dollars and call limit must be positive and finite")
        self.max_usd = max_usd
        self.max_calls = max_calls
        self._lock = asyncio.Lock()
        self._attempts = 0
        self._reservations: dict[int, float] = {}
        self._reported_usd = 0.0

    async def begin(self) -> int:
        """Bound preflights as well as generation attempts, including failures."""
        async with self._lock:
            if self._attempts >= self.max_calls:
                raise BudgetExceeded("model call limit reached")
            if self._reported_usd + sum(self._reservations.values()) >= self.max_usd:
                raise BudgetExceeded("model dollar budget reached")
            self._attempts += 1
            return self._attempts

    async def reserve(self, identifier: int, maximum_usd: float) -> None:
        if not math.isfinite(maximum_usd) or maximum_usd < 0:
            raise ValueError("reservation must be finite and nonnegative")
        async with self._lock:
            if identifier in self._reservations:
                raise ValueError("request already reserved")
            committed = self._reported_usd + sum(self._reservations.values())
            if committed + maximum_usd > self.max_usd:
                raise BudgetExceeded("request would exceed the remaining model dollar budget")
            self._reservations[identifier] = maximum_usd

    async def settle(self, identifier: int, reported_usd: float) -> None:
        if not math.isfinite(reported_usd) or reported_usd < 0:
            raise ValueError("reported spend must be finite and nonnegative")
        async with self._lock:
            self._reservations.pop(identifier)
            self._reported_usd += reported_usd

    def report(self) -> dict[str, Any]:
        return {
            "max_usd": self.max_usd,
            "max_calls": self.max_calls,
            "attempts": self._attempts,
            "reported_usd": self._reported_usd,
            "unsettled_reservations_usd": sum(self._reservations.values()),
            "unsettled_requests": len(self._reservations),
            "basis": "counted input + maximum output at standard token rates, 25% margin",
            "hosted_tools_allowed": False,
        }
