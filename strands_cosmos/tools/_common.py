# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Strands tool results and `just`-based execution.

Tool return format (@tool-compatible):
    {
        "status": "success" | "error",
        "content": [
            {"text": "..."},
            {"json": {...}},
            {"image": {"format": "jpeg", "source": {"bytes": b"..."}}},
        ],
    }

Design:
    All pipelines live in the top-level `justfile`. Python tools are *thin
    wrappers* that invoke `just <recipe> <args...>` and normalize the result.
    This keeps the agent, the operator's CLI, and the Cosmos upstream repos
    consistent with zero duplication.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


# ── ToolResult builders ──────────────────────────────────────────────────
def ok(
    text: str = "",
    data: dict | None = None,
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    image_format: str | None = None,
) -> dict:
    """Build a success ToolResult."""
    content: list[dict] = []
    if text:
        content.append({"text": text})
    if data is not None:
        content.append({"json": data})
    if image_bytes is not None:
        fmt = image_format or "jpeg"
        content.append({"image": {"format": fmt, "source": {"bytes": image_bytes}}})
    elif image_path:
        p = Path(image_path).expanduser()
        if p.exists():
            fmt = image_format or (p.suffix.lower().lstrip(".") or "jpeg")
            if fmt == "jpg":
                fmt = "jpeg"
            if fmt not in ("png", "jpeg", "gif", "webp"):
                fmt = "png"
            content.append({"image": {"format": fmt, "source": {"bytes": p.read_bytes()}}})
    if not content:
        content.append({"text": "ok"})
    return {"status": "success", "content": content}


def err(msg: str, data: dict | None = None) -> dict:
    """Build an error ToolResult."""
    content: list[dict] = [{"text": f"❌ {msg}"}]
    if data is not None:
        content.append({"json": data})
    return {"status": "error", "content": content}


# ── just runner ──────────────────────────────────────────────────────────
_JUST_BIN = os.getenv("STRANDS_COSMOS_JUST", "just")


def _find_justfile() -> Path | None:
    """Walk up from CWD to find the nearest justfile, else use package root."""
    p = Path.cwd()
    for parent in [p] + list(p.parents):
        f = parent / "justfile"
        if f.is_file():
            return f
    # Fall back to the bundled justfile shipped inside the package
    # (strands_cosmos/justfile — copied from the repo root at build time).
    pkg_dir = Path(__file__).resolve().parent.parent  # strands_cosmos/
    f = pkg_dir / "justfile"
    if f.is_file():
        return f
    # Editable/repo checkout: justfile at the repo root (parent of strands_cosmos/).
    f = pkg_dir.parent / "justfile"
    return f if f.is_file() else None


# Characters that allow breaking out of `just` {{param}} interpolation into the
# underlying shell or Python heredoc (the CWE-78 root cause). Any agent-reachable
# argument containing these is rejected at this single chokepoint as defense in
# depth -- agent-facing tools should additionally prefer the direct argv path
# (see _security.safe_run) over `just`.
_JUST_FORBIDDEN_CHARS = ("\"", "'", "`", ";", "\n", "\r", "$(", "${", "&&", "||", "|", ">", "<")


class JustInjectionError(ValueError):
    """Raised when a just_run argument contains shell/template metacharacters."""


def _reject_injection(args: tuple) -> str | None:
    """Return an error string if any arg contains a breakout metacharacter."""
    for a in args:
        s = str(a)
        for bad in _JUST_FORBIDDEN_CHARS:
            if bad in s:
                return (
                    f"refusing to pass argument containing {bad!r} to `just` "
                    f"(possible command/template injection): {s[:80]!r}"
                )
    return None


def just_run(
    recipe: str,
    *args: str,
    timeout_s: int = 3600,
    env: dict | None = None,
    cwd: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Invoke `just <recipe> <args...>` and return a normalized proc dict.

    Returns:
        {"ok": bool, "returncode": int, "stdout": str, "stderr": str, "cmd": str}
    """
    bad = _reject_injection(args)
    if bad is not None:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": bad,
            "cmd": f"{_JUST_BIN} {recipe} <rejected>",
        }

    if not shutil.which(_JUST_BIN):
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": f"`{_JUST_BIN}` not found on PATH. It ships with strands-cosmos (pip pkg `rust-just`); reinstall with `pip install -U strands-cosmos`, or `pip install rust-just`, or `brew install just`",
            "cmd": f"{_JUST_BIN} {recipe} " + " ".join(args),
        }

    justfile = _find_justfile()
    workdir = cwd or (str(justfile.parent) if justfile else None)

    cmd = [_JUST_BIN, recipe, *[str(a) for a in args]]
    run_env = os.environ.copy() if env is None else dict(env)
    if extra_env:
        run_env.update(extra_env)

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=workdir,
            env=run_env,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-8000:],
            "stderr": p.stderr[-4000:],
            "cmd": " ".join(cmd),
            "cwd": workdir,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"timeout after {timeout_s}s",
            "cmd": " ".join(cmd),
            "cwd": workdir,
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "cmd": " ".join(cmd),
            "cwd": workdir,
        }


def proc_result(proc: dict, success_text: str, fail_text: str = "") -> dict:
    """Convert a just_run output into a ToolResult."""
    if proc.get("ok"):
        tail = proc.get("stdout", "")[-15000:]
        return ok(
            text=success_text + (f"\n\n--- stdout (tail) ---\n{tail}" if tail else ""),
            data=proc,
        )
    stderr_tail = proc.get("stderr", "")[-4000:]
    return err(
        fail_text or f"exit {proc.get('returncode')}: {stderr_tail}",
        data=proc,
    )


# ── Legacy subprocess helper ─────────────────────────────────────────────
def run_proc(
    cmd: list[str],
    timeout_s: int = 3600,
    cwd: str | None = None,
    env: dict | None = None,
) -> dict:
    """Run a subprocess, capture output. Never raises."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            env=merged_env,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-80000:],
            "stderr": p.stderr[-40000:],
            "cmd": " ".join(cmd),
        }
    except FileNotFoundError as e:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": str(e), "cmd": " ".join(cmd)}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(e), "cmd": " ".join(cmd)}
