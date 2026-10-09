"""CodeQL py/insecure-temporary-file (CWE-377) regression for rtp_capture_frame.

The default output path must be created atomically (``tempfile.mkstemp``),
never merely *named* (``tempfile.mktemp``), and an empty placeholder must be
removed again when no frame was captured.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from strands_cosmos.tools import rtp


def test_rtp_module_does_not_use_mktemp():
    src = Path(rtp.__file__).read_text()
    assert "tempfile.mktemp(" not in src
    assert "tempfile.mkstemp(" in src


def test_rtp_default_output_is_created_atomically_and_cleaned(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "1")
    import tempfile
    tempfile.tempdir = None  # re-read TMPDIR
    seen: dict = {}

    def fake_just_run(recipe, *args, timeout_s=0, extra_env=None):
        out = Path(extra_env["RTP_OUTPUT"])
        seen["path"] = out
        # mkstemp semantics: file already exists, empty, owner-only perms.
        assert out.exists() and out.stat().st_size == 0
        assert out.name.startswith("cosmos_rtp_") and out.suffix == ".jpg"
        if os.name == "posix":
            assert (out.stat().st_mode & 0o777) == 0o600
        return {"returncode": 1, "stdout": "", "stderr": "no stream", "cmd": "fake"}

    monkeypatch.setattr(rtp, "just_run", fake_just_run)
    res = rtp.rtp_capture_frame(port=5004, timeout_s=1)
    assert res["status"] == "error"
    assert "path" in seen
    assert not seen["path"].exists(), "empty placeholder must be removed on failure"


def test_rtp_explicit_output_path_is_not_deleted_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    target = tmp_path / "frame.jpg"
    target.write_bytes(b"")

    monkeypatch.setattr(
        rtp, "just_run",
        lambda *a, **k: {"returncode": 1, "stdout": "", "stderr": "", "cmd": "fake"},
    )
    res = rtp.rtp_capture_frame(output_path=str(target), timeout_s=1)
    assert res["status"] == "error"
    assert target.exists(), "caller-provided paths are never unlinked"
