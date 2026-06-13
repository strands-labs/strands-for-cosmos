# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host summary via `just sysinfo`."""
from __future__ import annotations

from strands import tool
from ._common import just_run, ok, err


@tool
def cosmos_sysinfo() -> dict:
    """Host summary: OS, Jetson model, GPU, memory, thermal via `just sysinfo`.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries a text summary of the host (OS, Jetson model, GPU, memory, thermal); on error ``status`` is ``"error"`` with a message.
    """
    proc = just_run("sysinfo", timeout_s=10)
    if not proc.get("ok"):
        return err(f"sysinfo failed: {proc.get('stderr', '')[:200]}")
    return ok(
        text=proc.get("stdout", "").strip() or "no output",
        data={"cmd": proc.get("cmd"), "cwd": proc.get("cwd")},
    )
