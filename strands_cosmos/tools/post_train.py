# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-training (SFT / LoRA / RL) for Cosmos models via the ``just`` workflow.

Launches Cosmos-Reason2, Predict 2.5, or Transfer 2.5 fine-tuning jobs. Training
configs are read from inside the project workspace; the model family, strategy,
and GPU count are validated before the job is dispatched.
"""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, resolve_in_workspace, validate_identifier


_FAMILIES = {"reason2", "predict2_5", "transfer2_5"}
_STRATEGIES = {"full", "lora", "rl"}


@tool
def cosmos_post_train(
    config_path: str,
    model_family: str = "reason2",
    strategy: str = "full",
    num_gpus: int = 1,
    dry_run: bool = False,
) -> dict:
    """Launch a Cosmos post-training job via just.

    Supports:
      - reason2 (full|lora) → `just post-train-reason2 <config> <strategy>`
      - reason2 rl          → `just post-train-reason2-rl <config>`
      - predict2_5          → `just post-train-predict <config> <num_gpus>`
      - transfer2_5         → `just post-train-transfer <config> <num_gpus>`

    Args:
        config_path: YAML / TOML training config (inside the workspace).
        model_family: reason2 | predict2_5 | transfer2_5.
        strategy: full | lora | rl (rl is reason2 only).
        num_gpus: GPUs per node (predict/transfer only).
        dry_run: If True, just preview the recipe name.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the training job's launch output (or the recipe preview when ``dry_run`` is set); on error ``status`` is ``"error"`` with a message.
    """
    # Constrain enums first (positionally interpolated, CWE-78 defense).
    if model_family not in _FAMILIES:
        return err(f"unknown model_family: {model_family!r} (allowed: {sorted(_FAMILIES)})")
    if strategy not in _STRATEGIES:
        return err(f"invalid strategy: {strategy!r} (allowed: {sorted(_STRATEGIES)})")

    # Confine the (LLM-controlled) config path to the workspace, then assert it
    # carries no shell/template metacharacters before it is interpolated into
    # the recipe. resolve_in_workspace already rejects '..'/escape + non-existence.
    try:
        resolved = str(resolve_in_workspace(config_path, must_exist=True))
        config_path = validate_identifier(resolved, what="config_path")
    except SecurityError as e:
        return err(str(e))

    if model_family == "reason2" and strategy == "rl":
        recipe, args = "post-train-reason2-rl", (config_path,)
    elif model_family == "reason2":
        recipe, args = "post-train-reason2", (config_path, strategy)
    elif model_family == "predict2_5":
        recipe, args = "post-train-predict", (config_path, str(int(num_gpus)))
    else:  # transfer2_5
        recipe, args = "post-train-transfer", (config_path, str(int(num_gpus)))

    if dry_run:
        return proc_result(
            {"ok": True, "stdout": f"dry-run: just {recipe} " + " ".join(args),
             "returncode": 0, "cmd": f"just {recipe} {' '.join(args)}"},
            success_text=f"dry-run: just {recipe}",
        )

    proc = just_run(recipe, *args, timeout_s=60 * 60 * 12)
    return proc_result(
        proc,
        success_text=f"✅ post-training ({model_family}/{strategy}) finished",
        fail_text=f"post-training failed: {proc.get('stderr', '')[:300]}",
    )
