"""Offline checks for the small public model-layer surface."""

from commentary import llm


def test_public_model_layer_exports_current_adapters_and_helpers() -> None:
    assert {"ScriptedBackend", "openai_factory", "strict_schema"} <= set(llm.__all__)
    backend = llm.ScriptedBackend()
    assert callable(backend.parse)
    assert backend.total.cost_usd == 0
