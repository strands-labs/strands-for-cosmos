# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos 3 post-training (SFT) tools — thin wrappers over justfile `c3-train-*` recipes.

Supervised fine-tuning of Cosmos 3 models via the NVIDIA Cosmos Framework
(`cosmos_framework.scripts.train`). The framework is installed by
`just c3-setup-framework`. Full SFT is tested upstream on 8x H100 (80GB); on
smaller GPUs the convert / dataset / config-validation steps still run, and
the multi-GPU launch is documented.

Flow:
  1. cosmos3_train_convert            base checkpoint -> PyTorch DCP
     (or cosmos3_train_convert_vlm    for the reasoner VLM path)
  2. cosmos3_train_prep_dataset       captions JSONL -> SFT dataset JSONL  (optional)
  3. cosmos3_train                    run SFT (recipe -> paired launch shell)
  4. cosmos3_train_export             trained DCP -> HF safetensors

Recipes: vision_sft_nano | vision_sft_super | llava_ov | videophy2_nano
(use cosmos3_train_recipes to list what your framework checkout ships).
"""
from __future__ import annotations

from strands import tool

from ._common import just_run, proc_result
from ._security import SecurityError, validate_identifier

# Training is long-running; allow generous timeouts.
_CONVERT_TIMEOUT = 60 * 60        # 1h: checkpoint download + DCP conversion
_TRAIN_TIMEOUT = 60 * 60 * 24     # 24h: full SFT run
_QUICK_TIMEOUT = 60 * 10          # 10m: listing / config show / dataset prep


@tool
def cosmos3_train_recipes() -> dict:
    """List the Cosmos 3 SFT recipes + launch shells available in the framework checkout.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    proc = just_run("c3-train-recipes", timeout_s=_QUICK_TIMEOUT)
    return proc_result(proc, "cosmos3 SFT recipes:", "c3-train-recipes failed")


@tool
def cosmos3_train_show(recipe: str = "vision_sft_nano") -> dict:
    """Show / validate the resolved training config for an SFT recipe (no GPU needed).

    Args:
        recipe: SFT recipe name (e.g. vision_sft_nano, vision_sft_super,
            llava_ov, videophy2_nano).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        recipe = validate_identifier(recipe, what="recipe")
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("c3-train-show", recipe, timeout_s=_QUICK_TIMEOUT)
    return proc_result(proc, "cosmos3 train config (" + recipe + "):", "c3-train-show failed")


@tool
def cosmos3_train_convert(checkpoint: str = "nvidia/Cosmos3-Nano", out: str = "") -> dict:
    """Convert a base Cosmos 3 checkpoint to PyTorch DCP format for training.

    Args:
        checkpoint: Catalog name / HF id (e.g. Cosmos3-Nano, Cosmos3-Super).
        out: Output DCP dir (default examples/checkpoints/<name>).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        checkpoint = validate_identifier(checkpoint, what="checkpoint")
        out = validate_identifier(out, what="out", allow_empty=True)
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("c3-train-convert", checkpoint, out, timeout_s=_CONVERT_TIMEOUT)
    return proc_result(proc, "cosmos3 checkpoint -> DCP", "c3-train-convert failed")


@tool
def cosmos3_train_convert_vlm(
    checkpoint: str = "nvidia/Cosmos3-Nano",
    out: str = "examples/checkpoints/Cosmos3-Nano-VLM",
) -> dict:
    """Merge a Cosmos 3 LM onto the Qwen3-VL visual tower (reasoner VLM SFT path).

    Args:
        checkpoint: Base checkpoint (e.g. Cosmos3-Nano).
        out: Output VLM safetensors dir.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        checkpoint = validate_identifier(checkpoint, what="checkpoint")
        out = validate_identifier(out, what="out", allow_empty=True)
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("c3-train-convert-vlm", checkpoint, out, timeout_s=_CONVERT_TIMEOUT)
    return proc_result(proc, "cosmos3 VLM checkpoint -> " + out, "c3-train-convert-vlm failed")


@tool
def cosmos3_train_prep_dataset(captions: str, out: str) -> dict:
    """Convert a captions JSONL into an SFT dataset JSONL.

    Args:
        captions: Path to the input captions JSONL.
        out: Path to write the SFT dataset JSONL.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        captions = validate_identifier(captions, what="captions")
        out = validate_identifier(out, what="out")
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("c3-train-prep-dataset", captions, out, timeout_s=_QUICK_TIMEOUT)
    return proc_result(proc, "cosmos3 SFT dataset -> " + out, "c3-train-prep-dataset failed")


@tool
def cosmos3_train(
    recipe: str = "vision_sft_nano",
    nproc: int = 8,
    dataset: str = "",
    checkpoint: str = "",
    overrides: str = "",
) -> dict:
    """Run Cosmos 3 supervised fine-tuning via the paired launch shell.

    Tested upstream on 8x H100 (80GB). Provide a smaller `nproc` (and Hydra
    `overrides` like "trainer.max_iter=10 optimizer.lr=1e-5") for short smokes.

    Args:
        recipe: SFT recipe (vision_sft_nano | vision_sft_super | llava_ov | videophy2_nano).
        nproc: GPUs for torchrun --nproc_per_node (default 8).
        dataset: Override DATASET_PATH (default: recipe's examples/data path).
        checkpoint: Override BASE_CHECKPOINT_PATH (the DCP dir from convert step).
        overrides: Space-separated Hydra tail overrides applied after `--`.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        recipe = validate_identifier(recipe, what="recipe")
        dataset = validate_identifier(dataset, what="dataset", allow_empty=True)
        checkpoint = validate_identifier(checkpoint, what="checkpoint", allow_empty=True)
        overrides = validate_identifier(overrides, what="overrides", allow_empty=True)
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run(
        "c3-train", recipe, str(nproc), dataset, checkpoint, overrides,
        timeout_s=_TRAIN_TIMEOUT,
    )
    return proc_result(proc, "cosmos3 SFT (" + recipe + ") finished", "c3-train failed")


@tool
def cosmos3_train_export(run_dir: str, out: str = "") -> dict:
    """Export a trained Cosmos 3 DCP checkpoint to HuggingFace safetensors.

    Args:
        run_dir: Training run dir ($IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>).
        out: Output dir (default <run_dir>/hf_export).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        run_dir = validate_identifier(run_dir, what="run_dir")
        out = validate_identifier(out, what="out", allow_empty=True)
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    proc = just_run("c3-train-export", run_dir, out, timeout_s=_CONVERT_TIMEOUT)
    return proc_result(proc, "cosmos3 trained checkpoint -> HF safetensors", "c3-train-export failed")
