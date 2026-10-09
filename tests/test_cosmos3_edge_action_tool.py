"""cosmos3_action_edge: argument validation and output plumbing with a fake pipeline (no GPU)."""
import json
import sys
import types

import pytest

from strands_cosmos.tools import cosmos3_edge as t


def _text(res):
    return " ".join(c.get("text", "") for c in res["content"])


@pytest.fixture
def ws(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    from PIL import Image
    Image.new("RGB", (64, 48), "red").save(tmp_path / "frame.png")
    return tmp_path


def test_unknown_and_untrained_embodiments_fail_before_any_load(ws, monkeypatch):
    monkeypatch.setattr(t, "_get_pipeline", lambda *_: pytest.fail("pipeline must not load"))
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), embodiment="so101")
    assert r["status"] == "error" and "not a Cosmos3-Edge embodiment" in _text(r)
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), embodiment="pusht")
    assert r["status"] == "error" and "untrained" in _text(r)


def test_bad_mode_view_and_path(ws, monkeypatch):
    monkeypatch.setattr(t, "_get_pipeline", lambda *_: pytest.fail("pipeline must not load"))
    assert t.cosmos3_action_edge(image=str(ws / "frame.png"), mode="dream")["status"] == "error"
    assert t.cosmos3_action_edge(image=str(ws / "frame.png"), view_point="drone")["status"] == "error"
    r = t.cosmos3_action_edge(image="/etc/passwd")
    assert r["status"] == "error" and "workspace" in _text(r)


def test_forward_dynamics_needs_matching_raw_actions(ws, monkeypatch):
    monkeypatch.setattr(t, "_get_pipeline", lambda *_: pytest.fail("pipeline must not load"))
    _fake_diffusers(monkeypatch)
    out = str(ws / "fd")
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), mode="forward_dynamics", out=out)
    assert "raw_actions_json" in _text(r)
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), mode="forward_dynamics", out=out,
                              raw_actions_json=json.dumps([[0.0] * 7] * 4))
    assert r["status"] == "error" and "[T][10]" in _text(r)


def _fake_diffusers(monkeypatch):
    import torch
    mod = types.ModuleType("diffusers.pipelines.cosmos.pipeline_cosmos3_omni")

    class CosmosActionCondition:
        def __init__(self, **kw):
            self.kw = kw
    mod.CosmosActionCondition = CosmosActionCondition
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.cosmos.pipeline_cosmos3_omni", mod)
    return torch


def test_policy_happy_path_with_fake_pipeline(ws, monkeypatch):
    torch = _fake_diffusers(monkeypatch)
    from PIL import Image
    seen = {}

    class FakePipe:
        device = "cpu"

        def __call__(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(action=[torch.ones(16, 10) * 0.5], video=[Image.new("RGB", (8, 8))] * 17)

    monkeypatch.setattr(t, "_get_pipeline", lambda mid: seen.setdefault("model_id", mid) and FakePipe())
    out = ws / "run1"
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), prompt="pick the cube", embodiment="droid_lerobot",
                              out=str(out), steps=3)
    assert r["status"] == "success", _text(r)
    assert seen["model_id"] == "nvidia/Cosmos3-Edge"
    cond = seen["action"].kw
    assert cond["domain_name"] == "droid_lerobot" and cond["mode"] == "policy" and cond["chunk_size"] == 16
    assert seen["num_inference_steps"] == 3
    data = json.loads((out / "actions.json").read_text())
    assert data["domain_id"] == 8 and len(data["actions"]) == 16 and len(data["actions"][0]) == 10
    assert len(list(out.glob("frame_*.png"))) == 17
    assert "actions [16, 10]" in _text(r)


def test_policy_droid_checkpoint_passes_validation_and_reaches_load(ws, monkeypatch):
    def boom(mid):
        raise RuntimeError(f"would load {mid}")
    monkeypatch.setattr(t, "_get_pipeline", boom)
    _fake_diffusers(monkeypatch)
    # catalog: the DROID policy checkpoints have an action surface -> validation passes, load is reached
    r = t.cosmos3_action_edge(image=str(ws / "frame.png"), model="edge-policy-droid", embodiment="bridge_orig_lerobot",
                              out=str(ws / "p"))
    assert r["status"] == "error" and "would load nvidia/Cosmos3-Edge-Policy-DROID" in _text(r)
