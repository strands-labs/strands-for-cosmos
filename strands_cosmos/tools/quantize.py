# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just quantize` — FP8/INT8/INT4 quantization (x86)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result
from ._security import SecurityError, validate_identifier


@tool
def cosmos_quantize(
    model_dir: str = "nvidia/Cosmos-Reason2-2B",
    output_dir: str = "./quantized/Cosmos-Reason2-2B-fp8",
    dtype: str = "fp16",
    quantization: str = "fp8",
) -> dict:
    """Quantize a Cosmos VLM/LLM via `just quantize` (x86 GPU host only).

    Args:
        model_dir: HF model id or local path.
        output_dir: Where to write quantized weights.
        dtype: Base precision (fp16 | bf16).
        quantization: Target quantization (fp8 | int8 | int4).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the quantized model weights; on error ``status`` is ``"error"`` with a message.
    """
    try:
        model_dir = validate_identifier(model_dir, what="model_dir")
        output_dir = validate_identifier(output_dir, what="output_dir")
        dtype = validate_identifier(dtype, what="dtype")
        quantization = validate_identifier(quantization, what="quantization", allow_empty=True)
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("quantize", model_dir, output_dir, dtype, quantization, timeout_s=60 * 60 * 3)
    return proc_result(
        proc,
        success_text=f"✅ quantized {model_dir} ({dtype}→{quantization}) → {output_dir}",
        fail_text=f"quantization failed: {proc.get('stderr', '')[:200]}",
    )
