"""Check the real saved handoff and its evidence boundaries without paid calls."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "examples/graphon-commentary"


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = runpy.run_path(str(ROOT / "scripts/prepare_graphon_commentary.py"))
    source = json.loads((SAVED / "sources.json").read_text())
    retrieval = json.loads((SAVED / "retrieval.json").read_text())
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc in source["documents"]:
        text = (
            f"Title: {doc['title']}\nSource: {doc['url']}\n"
            f"Published: {doc['published_at']}\n\n{doc['excerpt']}"
        )
        (corpus / doc["key"]).write_text(text)
    module["candidates"].__globals__["OUT"] = tmp_path
    return module, source, retrieval


def test_selected_quote_is_copied_from_actual_graphon_read_and_only_quote_reaches_writer(archive):
    module, sources, retrieval = archive
    candidates = module["candidates"](retrieval, sources)
    writer = json.loads((SAVED / "writer-input.json").read_text())
    selected = next(p for p in candidates if p["id"] == writer["fact"]["id"])
    assert writer["fact"]["text"] == selected["text"]
    assert selected["text"] in selected["retrieved_text"]
    assert selected["span"] in retrieval["search"]["results"][1]["spans"]
    assert set(writer) == {"event", "fact"}
    assert "Arsenal" in writer["fact"]["text"]
    assert not any(key in writer for key in ("reference", "target_commentary", "review"))
    assert all(p["published_at"] <= "2025-11-08T17:30:00" for p in candidates)


def test_future_source_and_changed_version_are_rejected(archive):
    module, sources, retrieval = archive
    future = copy.deepcopy(sources)
    future["documents"][0]["published_at"] = "2025-11-09T09:00:00Z"
    with pytest.raises(ValueError, match="before kick-off"):
        module["candidates"](retrieval, future)
    retrieval["read"]["results"][0]["span"]["versionId"] = "different-version"
    with pytest.raises(ValueError, match="changed Graphon source"):
        module["candidates"](retrieval, sources)


def test_modified_text_and_incomplete_reads_are_rejected(archive):
    module, sources, retrieval = archive
    retrieval["read"]["results"][0]["text"] += " invented evidence"
    with pytest.raises(ValueError, match="character slice"):
        module["candidates"](retrieval, sources)
    retrieval["read"]["next"] = {"spans": []}
    with pytest.raises(ValueError, match="incomplete"):
        module["candidates"](retrieval, sources)


def test_public_commentary_is_exact_model_output_and_audited_speech():
    raw = json.loads((SAVED / "graphon-writer.json").read_text())
    message = next(item for item in raw["output"] if item["type"] == "message")
    output = json.loads(
        next(item["text"] for item in message["content"] if item["type"] == "output_text")
    )
    public = json.loads((ROOT / "web/public/graphon-commentary/run.json").read_text())
    assert output == public["commentary"]
    spoken = next(line for line in public["delivery"] if line["voice"] == "analyst")
    assert spoken["text"] == output["text"]
    assert all(line["speed"] == 1 for line in public["delivery"])
    assert public["budget"]["attempts"] == 2
    assert public["budget"]["unsettled_requests"] == 0
    assert public["budget"]["reported_usd"] < 0.03
    assert public["tts_characters"] <= 240
