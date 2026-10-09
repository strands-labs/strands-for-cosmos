"""Cosmos3EdgeHFModel: construction, config and <think> routing — no weights, no GPU."""
import asyncio

import pytest

from strands_cosmos import cosmos3_edge_hf_model as mod
from strands_cosmos.cosmos3_edge_hf_model import Cosmos3EdgeHFModel, _route_thinking


def test_defaults_and_aliases():
    m = Cosmos3EdgeHFModel()
    cfg = m.get_config()
    assert cfg["model_id"] == "nvidia/Cosmos3-Edge"
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["show_thinking"] is False
    assert Cosmos3EdgeHFModel(model_id="edge").get_config()["model_id"] == "nvidia/Cosmos3-Edge"
    assert m.model is None  # lazy


def test_nano_is_refused_with_pointer_to_vllm():
    with pytest.raises(ValueError, match="Cosmos3ReasonerModel"):
        Cosmos3EdgeHFModel(model_id="nvidia/Cosmos3-Nano")


def test_local_finetune_path_is_accepted():
    m = Cosmos3EdgeHFModel(model_id="/data/my-edge-sft")
    assert m.get_config()["model_id"] == "/data/my-edge-sft"


def test_version_gate(monkeypatch):
    assert mod._version_at_least("5.19.0", "5.19.0")
    assert mod._version_at_least("5.20.0.dev0", "5.19.0")
    assert not mod._version_at_least("5.2.0", "5.19.0")


def test_load_errors_clearly_on_old_transformers(monkeypatch):
    import types, sys
    fake = types.SimpleNamespace(__version__="5.2.0")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    m = Cosmos3EdgeHFModel()
    with pytest.raises(ImportError, match="transformers>=5.19.0"):
        m._load_model()


async def _agen(chunks):
    yield {"messageStart": {"role": "assistant"}}
    yield {"contentBlockStart": {"start": {}}}
    for c in chunks:
        yield {"contentBlockDelta": {"delta": {"text": c}}}
    yield {"contentBlockStop": {}}
    yield {"messageStop": {"stopReason": "end_turn"}}


def _collect(chunks):
    async def run():
        out = []
        async for ev in _route_thinking(_agen(chunks)):
            out.append(ev)
        return out
    return asyncio.run(run())


def _texts(events, key):
    vals = []
    for ev in events:
        d = ev.get("contentBlockDelta", {}).get("delta", {})
        if key == "text" and "text" in d:
            vals.append(d["text"])
        if key == "reasoning" and "reasoningContent" in d:
            vals.append(d["reasoningContent"]["text"])
    return "".join(vals)


def test_template_opened_think_block_is_routed_to_reasoning():
    # exactly what Edge emits: thinking first, no opening tag, then </think>, then the answer
    chunks = ["The image is a solid", " red square.", "\n</thi", "nk>\n\nA solid red", " square."]
    ev = _collect(chunks)
    assert _texts(ev, "reasoning") == "The image is a solid red square.\n"
    assert _texts(ev, "text") == "A solid red square."


def test_explicit_think_tag_streams_incrementally():
    chunks = ["<think>", "step one,", " step two", "</think>", "answer"]
    ev = _collect(chunks)
    assert _texts(ev, "reasoning") == "step one, step two"
    assert _texts(ev, "text") == "answer"
    # at least one reasoning delta was emitted before the closing tag arrived
    kinds = ["r" if "reasoningContent" in e.get("contentBlockDelta", {}).get("delta", {}) else
             "t" if "text" in e.get("contentBlockDelta", {}).get("delta", {}) else "-" for e in ev]
    assert kinds.index("r") < kinds.index("t")


def test_no_think_block_at_all_is_plain_text_and_nothing_is_lost():
    chunks = ["Hello", " there", ", robot."]
    ev = _collect(chunks)
    assert _texts(ev, "reasoning") == ""
    assert _texts(ev, "text") == "Hello there, robot."
    # structure preserved
    assert [k for e in ev for k in e][0] == "messageStart"
    assert "messageStop" in ev[-1]


def test_show_thinking_passthrough(monkeypatch):
    async def fake_parent(*a, **k):
        for ev in [{"contentBlockDelta": {"delta": {"text": "x</think>y"}}}]:
            yield ev
    monkeypatch.setattr(mod.CosmosVisionModel, "stream", lambda self, *a, **k: fake_parent())
    m = Cosmos3EdgeHFModel(show_thinking=True)
    m.model = object()  # pretend loaded; stream() must not try to load

    async def run():
        return [e async for e in m.stream([{"role": "user", "content": [{"text": "hi"}]}])]
    out = asyncio.run(run())
    assert out == [{"contentBlockDelta": {"delta": {"text": "x</think>y"}}}]
