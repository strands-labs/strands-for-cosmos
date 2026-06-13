# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just export-llm` / `just export-visual` (x86)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, validate_identifier


@tool
def cosmos_export_onnx(
    model_dir: str,
    output_dir: str,
    which_part: str = "llm",
    dtype: str = "fp16",
    quantization: str = "",
) -> dict:
    """Export a Cosmos model component to ONNX via just recipes.

    Args:
        model_dir: Path to model (quantized for llm, HF original for visual).
        output_dir: Destination for .onnx files.
        which_part: "llm" | "visual".
        dtype: Base dtype (visual only).
        quantization: e.g. "fp8" (visual only).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the exported ONNX model; on error ``status`` is ``"error"`` with a message.
    """
    try:
        model_dir = validate_identifier(model_dir, what="model_dir")
        output_dir = validate_identifier(output_dir, what="output_dir")
        dtype = validate_identifier(dtype, what="dtype", allow_empty=True)
        quantization = validate_identifier(quantization, what="quantization", allow_empty=True)
    except SecurityError as e:
        return err(str(e))
    if which_part == "llm":
        proc = just_run("export-llm", model_dir, output_dir, timeout_s=60 * 60 * 2)
    elif which_part == "visual":
        proc = just_run("export-visual", model_dir, output_dir, dtype, quantization,
                        timeout_s=60 * 60 * 2)
    else:
        return err(f"which_part must be 'llm' or 'visual', got {which_part!r}")

    return proc_result(
        proc,
        success_text=f"✅ ONNX ({which_part}) exported → {output_dir}",
        fail_text=f"ONNX export failed: {proc.get('stderr', '')[:200]}",
    )
