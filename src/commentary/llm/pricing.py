"""Dollars per million tokens, so every run reports what it cost."""

from __future__ import annotations

from commentary.llm.base import Usage

#: (input, output) USD per million tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def cost_usd(model: str, usage: Usage) -> float:
    """What one call cost. Unknown models price at zero rather than lying."""
    rates = PRICES.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    million = 1_000_000
    return (
        usage.input_tokens * in_rate
        + usage.cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * out_rate
    ) / million
