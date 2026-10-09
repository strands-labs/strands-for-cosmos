"""Tool-level plumbing for Cosmos3-Edge (no GPU, no weights): sound refusal + reasoner backend."""
import pytest

from strands_cosmos.tools import cosmos3 as t


def _text(res):
    return " ".join(c.get("text", "") for c in res["content"])


def test_sound_tools_refuse_edge_before_loading_anything(monkeypatch):
    monkeypatch.setenv("C3_MODEL", "nvidia/Cosmos3-Edge")
    monkeypatch.delenv("C3_GEN_MODEL", raising=False)
    calls = []
    monkeypatch.setattr(t, "just_run", lambda *a, **k: calls.append(a) or {"returncode": 0, "stdout": "", "stderr": ""})
    res = t.cosmos3_text2video_sound(prompt="rain on a tin roof")
    assert res["status"] == "error"
    assert "no 'sound' surface" in _text(res) and "Cosmos3-Nano" in _text(res)
    assert calls == [], "the recipe must not run when the checkpoint cannot do sound"


def test_c3_gen_model_wins_over_c3_model(monkeypatch):
    monkeypatch.setenv("C3_MODEL", "nvidia/Cosmos3-Edge")
    monkeypatch.setenv("C3_GEN_MODEL", "nvidia/Cosmos3-Nano")
    assert t._sound_unsupported("text2video-with-sound") is None


def test_silent_modes_are_not_blocked_on_edge(monkeypatch):
    monkeypatch.setenv("C3_MODEL", "edge")
    assert t._sound_unsupported("text2video") is None
    assert t._sound_unsupported("text2image") is None


def test_reason_backend_hf_uses_in_process_edge_model(monkeypatch):
    seen = {}

    class FakeModel:
        config = {"model_id": "nvidia/Cosmos3-Edge"}

        def update_config(self, **kw):
            seen["cfg"] = kw

        async def stream(self, messages):
            seen["messages"] = messages
            yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}}
            yield {"contentBlockDelta": {"delta": {"text": "a red square"}}}

    monkeypatch.setattr(t, "_get_hf_reasoner", lambda model: FakeModel())
    res = t.cosmos3_reason(prompt="Caption:", image="ws_red.png", backend="hf")
    assert res["status"] == "success"
    assert "a red square" in _text(res) and "thinking" not in _text(res)
    # media handed to the HF path as tags (it resolves + confines paths itself)
    assert seen["messages"][0]["content"] == [{"text": "<image>ws_red.png</image> Caption:"}]
    assert seen["cfg"]["params"]["max_tokens"] == 4096


def test_reason_bad_backend_is_an_error_not_a_crash():
    res = t.cosmos3_reason(prompt="x", backend="grpc")
    assert res["status"] == "error"
    assert "backend" in _text(res)


def test_hf_reasoner_cache_resolves_aliases(monkeypatch):
    created = []

    class Fake:
        def __init__(self, model_id):
            created.append(model_id)

    import strands_cosmos.cosmos3_edge_hf_model as m
    monkeypatch.setattr(m, "Cosmos3EdgeHFModel", Fake)
    t._hf_models.clear()
    a = t._get_hf_reasoner("edge")
    b = t._get_hf_reasoner("nvidia/Cosmos3-Edge")
    assert a is b and created == ["nvidia/Cosmos3-Edge"]
