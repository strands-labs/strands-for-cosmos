# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just serve-*` recipes (TRT-Edge-LLM server)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, validate_identifier


@tool
def cosmos_serve(
    action: str = "status",
    llm_engine_dir: str = "",
    visual_engine_dir: str = "",
    port: int = 8080,
    host: str = "127.0.0.1",
    lines: int = 80,
) -> dict:
    """Manage the TRT-Edge-LLM inference server via just recipes.

    Args:
        action: "start" | "stop" | "restart" | "status" | "logs".
        llm_engine_dir / visual_engine_dir: required for start/restart.
        port / host: bind address for start/restart.
        lines: number of log lines (for action=logs).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the inference server's start/stop/status output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        if llm_engine_dir:
            llm_engine_dir = validate_identifier(llm_engine_dir, what="llm_engine_dir")
        if visual_engine_dir:
            visual_engine_dir = validate_identifier(visual_engine_dir, what="visual_engine_dir")
        host = validate_identifier(host, what="host")
    except SecurityError as e:
        return err(str(e))
    if action == "start":
        if not llm_engine_dir or not visual_engine_dir:
            return err("llm_engine_dir and visual_engine_dir are required for start")
        proc = just_run("serve-start", llm_engine_dir, visual_engine_dir,
                        str(port), host, timeout_s=30)
        return proc_result(proc, success_text=f"▶ serve-start → http://{host}:{port}")
    if action == "restart":
        if not llm_engine_dir or not visual_engine_dir:
            return err("llm_engine_dir and visual_engine_dir are required for restart")
        proc = just_run("serve-restart", llm_engine_dir, visual_engine_dir, timeout_s=30)
        return proc_result(proc, success_text="▶ serve-restart done")
    if action == "stop":
        proc = just_run("serve-stop", timeout_s=10)
        return proc_result(proc, success_text="⏹ server stopped")
    if action == "status":
        proc = just_run("serve-status", timeout_s=5)
        return proc_result(proc, success_text="server status")
    if action == "logs":
        proc = just_run("serve-logs", str(lines), timeout_s=5)
        return proc_result(proc, success_text=f"server logs (last {lines} lines)")
    return err(f"unknown action: {action} (valid: start|stop|restart|status|logs)")
