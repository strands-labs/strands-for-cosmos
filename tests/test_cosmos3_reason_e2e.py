"""End-to-end test: cosmos3 reasoner *tools* drive the SDK model provider.

Proves the architectural fix (tools use `Cosmos3ReasonerModel`, not a `just`
shell-out) AND the security win: a traversal path in a `<video>` tag is dropped
by the provider's workspace-confined media resolver, so no `file://<abspath>`
arbitrary-read request ever reaches the backend (the old recipe's CWE-22 sink).

Runs without a GPU/vLLM server by stubbing the OpenAI client the provider uses.
"""
import pytest

from strands_cosmos.tools import cosmos3


class _FakeDelta:
    def __init__(self, content): self.content = content

class _FakeChoice:
    def __init__(self, content): self.delta = _FakeDelta(content)

class _FakeChunk:
    def __init__(self, content): self.choices = [_FakeChoice(content)]

class _FakeCompletions:
    """Captures the request and streams a canned reply."""
    last_request = None
    def create(self, **kwargs):
        _FakeCompletions.last_request = kwargs
        return iter([_FakeChunk("a "), _FakeChunk("dog "), _FakeChunk("runs.")])

class _FakeModels:
    def list(self):
        class _M: id = "nvidia/Cosmos3-Nano"
        class _R: data = [_M()]
        return _R()

class _FakeClient:
    def __init__(self): self.chat = type("C", (), {"completions": _FakeCompletions()})()
    @property
    def models(self): return _FakeModels()


@pytest.fixture
def fake_server(monkeypatch):
    cosmos3._reasoner_models.clear()
    _FakeCompletions.last_request = None
    from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel
    monkeypatch.setattr(Cosmos3ReasonerModel, "_get_client", lambda self: _FakeClient())
    monkeypatch.setattr(Cosmos3ReasonerModel, "_resolve_model_id",
                        lambda self: "nvidia/Cosmos3-Nano")
    return _FakeCompletions


def test_reason_tool_drives_sdk_provider(fake_server):
    """cosmos3_reason returns text streamed via the SDK provider (no shell-out)."""
    res = cosmos3.cosmos3_reason(prompt="What happens?", think=False)
    assert res["status"] == "success"
    assert "dog" in res["content"][0]["text"]
    # the request went through the OpenAI client, proving SDK-provider path
    assert fake_server.last_request is not None
    assert fake_server.last_request["messages"][-1]["role"] == "user"


def test_reason_tool_drops_traversal_video(fake_server, monkeypatch, tmp_path):
    """A <video>/etc/passwd</video> traversal is confined away by the provider.

    The old `just c3-reason` recipe would have built
    `file:///etc/passwd` and handed it to the backend (CWE-22). The SDK
    provider's `_media_to_url` raises SecurityError on the out-of-workspace
    path, so the resolver drops it: NO file:// url, NO /etc/passwd in payload.
    """
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    res = cosmos3.cosmos3_caption(video="/etc/passwd")
    # Tool surfaces the SecurityError as an error result (fail-closed).
    assert res["status"] == "error"
    # And crucially: nothing resembling a file:// passwd read was sent.
    req = fake_server.last_request
    if req is not None:
        payload = str(req)
        assert "file://" not in payload
        assert "/etc/passwd" not in payload


def test_reason_tool_confines_but_allows_workspace_video(fake_server, monkeypatch, tmp_path):
    """A real in-workspace video is base64-embedded (no file:// path leak)."""
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    res = cosmos3.cosmos3_caption(video=str(vid))
    assert res["status"] == "success"
    req = fake_server.last_request
    payload = str(req)
    # confined + embedded as data URI, never as a raw file path/URL
    assert "file://" not in payload
    assert "data:" in payload


# ── Native ContentBlock shape (D2/D5): tool emits image/video blocks ──────

def test_tool_emits_native_video_block_not_tag(fake_server, monkeypatch, tmp_path):
    """Video must travel as a native {"video": {...}} block → video_url payload.

    Regression guard against the old `<video>path</video>` tag-string shape.
    """
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    vid = tmp_path / "scene.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    res = cosmos3.cosmos3_caption(video=str(vid))
    assert res["status"] == "success"
    req = fake_server.last_request
    user = req["messages"][-1]
    # content is a list of typed parts; a video_url part must be present
    parts = user["content"]
    assert isinstance(parts, list)
    kinds = [p.get("type") for p in parts]
    assert "video_url" in kinds, f"expected native video_url part, got {kinds}"
    # never the old tag string smuggled in text
    text_parts = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
    assert "<video>" not in text_parts


def test_media_format_inferred_from_extension():
    """Extension → Strands media `format` literal (mov/webp/etc.)."""
    assert cosmos3._media_format("/x/clip.mov", "mp4") == "mov"
    assert cosmos3._media_format("/x/frame.JPG", "png") == "jpeg"
    assert cosmos3._media_format("https://h/v.webm?sig=1", "mp4") == "webm"
    assert cosmos3._media_format("/x/noext", "mp4") == "mp4"  # fallback


def test_native_video_traversal_still_blocked(fake_server, monkeypatch, tmp_path):
    """Native video block with a traversal path is still confined (fail-closed)."""
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    res = cosmos3.cosmos3_temporal(video="../../../../etc/passwd")
    assert res["status"] == "error"
    req = fake_server.last_request
    if req is not None:
        payload = str(req)
        assert "file://" not in payload
        assert "/etc/passwd" not in payload
