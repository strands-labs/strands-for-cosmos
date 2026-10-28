"""The agent tool and the generator agree on which resolutions exist."""
import pytest

from strands_cosmos.cosmos3_generator_model import RES_TIERS, Cosmos3GeneratorModel
from strands_cosmos.tools import cosmos3 as c3tools


def test_tool_gate_matches_model_tiers():
    assert set(c3tools._GEN_RES) == set(RES_TIERS)


def test_tool_accepts_every_model_tier(monkeypatch):
    calls = []
    monkeypatch.setattr(c3tools, "just_run", lambda *a, **k: calls.append(a) or {"ok": True})
    for res in RES_TIERS:
        c3tools._gen("text2image", "p", "", "/tmp/x.png", 1, 24, 1, 6.0, res, "false", 0)
    assert [a[6] for a in calls] == list(RES_TIERS)


def test_tool_refuses_unsupported_resolution(monkeypatch):
    monkeypatch.setattr(c3tools, "just_run", lambda *a, **k: pytest.fail("must not run"))
    out = c3tools._gen("text2image", "p", "", "/tmp/x.png", 1, 24, 1, 6.0, "1080", "false", 0)
    assert out["status"] == "error"
    assert "1080" in out["content"][0]["text"]


def test_model_refuses_unsupported_resolution_without_loading():
    m = Cosmos3GeneratorModel()
    m._load_pipeline = lambda: pytest.fail("must refuse before loading the pipeline")
    with pytest.raises(ValueError, match="1080"):
        m.generate(mode="text2image", prompt="p", resolution="1080")
