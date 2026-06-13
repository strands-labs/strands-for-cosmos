# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just build-llm-engine` / `just build-visual-engine` (Thor)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, validate_identifier


@tool
def cosmos_build_engine(
    onnx_dir: str,
    engine_dir: str,
    which_part: str = "llm",
    min_image_tokens: int = 4,
    max_image_tokens: int = 10240,
    max_input_len: int = 1024,
) -> dict:
    """Build a TensorRT engine from ONNX on Jetson Thor.

    Args:
        onnx_dir: Directory of exported ONNX files.
        engine_dir: Destination for built `.engine` files.
        which_part: "llm" | "visual".
        min_image_tokens / max_image_tokens / max_input_len: LLM engine hyper-params.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the build log and resulting TensorRT engine path; on error ``status`` is ``"error"`` with a message.
    """
    try:
        onnx_dir = validate_identifier(onnx_dir, what="onnx_dir")
        engine_dir = validate_identifier(engine_dir, what="engine_dir")
    except SecurityError as e:
        return err(str(e))
    if which_part == "llm":
        proc = just_run(
            "build-llm-engine", onnx_dir, engine_dir,
            str(min_image_tokens), str(max_image_tokens), str(max_input_len),
            timeout_s=60 * 60 * 2,
        )
    elif which_part == "visual":
        proc = just_run("build-visual-engine", onnx_dir, engine_dir, timeout_s=60 * 60 * 2)
    else:
        return err(f"which_part must be 'llm' or 'visual', got {which_part!r}")

    return proc_result(
        proc,
        success_text=f"✅ TRT engine ({which_part}) built → {engine_dir}",
        fail_text=f"engine build failed: {proc.get('stderr', '')[:200]}",
    )
