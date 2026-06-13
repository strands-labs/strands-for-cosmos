# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just distill` (KD / DMD2)."""
from __future__ import annotations

from pathlib import Path

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, validate_identifier


@tool
def cosmos_distill(
    teacher_checkpoint: str,
    student_output: str,
    method: str = "kd",
    model_family: str = "transfer2_5",
    num_gpus: int = 8,
) -> dict:
    """Distill a Teacher model into a faster Student via `just distill`.

    Args:
        teacher_checkpoint: Path to Teacher checkpoint.
        student_output: Destination for Student checkpoint.
        method: kd | dmd2.
        model_family: transfer2_5 | predict2_5.
        num_gpus: GPUs per node.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the distilled student checkpoint; on error ``status`` is ``"error"`` with a message.
    """
    if not Path(teacher_checkpoint).exists():
        return err(f"teacher checkpoint not found: {teacher_checkpoint}")
    if method not in {"kd", "dmd2"}:
        return err(f"method must be 'kd' or 'dmd2', got {method!r}")
    if model_family not in {"transfer2_5", "predict2_5"}:
        return err(f"model_family must be transfer2_5 or predict2_5, got {model_family!r}")

    try:
        teacher_checkpoint = validate_identifier(teacher_checkpoint, what="teacher_checkpoint")
        student_output = validate_identifier(student_output, what="student_output")
    except SecurityError as e:
        return err(str(e))

    Path(student_output).parent.mkdir(parents=True, exist_ok=True)

    proc = just_run(
        "distill",
        teacher_checkpoint, student_output, method, model_family, str(num_gpus),
        timeout_s=60 * 60 * 24,
    )
    return proc_result(
        proc,
        success_text=f"✅ distillation ({method}) → {student_output}",
        fail_text=f"distillation failed: {proc.get('stderr', '')[:300]}",
    )
