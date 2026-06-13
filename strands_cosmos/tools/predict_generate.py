# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just predict-generate` (Cosmos-Predict 2.5)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from strands import tool
from ._common import just_run, proc_result


@tool
def cosmos_predict_generate(
    prompt: str,
    output_dir: str = "./outputs/predict2_5",
    input_image: str = "",
    input_video: str = "",
    num_frames: int = 121,
    height: int = 720,
    width: int = 1280,
    fps: int = 24,
    guidance_scale: float = 7.0,
    num_steps: int = 35,
    seed: int = 0,
    checkpoint: str = "",
    model_variant: str = "video2world",
    repo_dir: str = "",
) -> dict:
    """Generate video with Cosmos-Predict2.5 (world model).

    Variants: text2world | video2world | action_conditioned | multiview.

    Args:
        prompt: Text description of the scene / action.
        output_dir: Output directory for generated .mp4.
        input_image / input_video: Seed inputs (video2world).
        num_frames / height / width / fps: Output geometry.
        guidance_scale / num_steps: Diffusion params.
        seed: Random seed.
        checkpoint: Optional fine-tuned checkpoint.
        model_variant: text2world | video2world | action_conditioned | multiview.
        repo_dir: Override COSMOS_PREDICT_REPO.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the generated video; on error ``status`` is ``"error"`` with a message.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params = {
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps,
        "seed": seed,
        "output_dir": output_dir,
        "variant": model_variant,
    }
    if input_image:
        params["input_image"] = input_image
    if input_video:
        params["input_video"] = input_video
    if checkpoint:
        params["checkpoint"] = checkpoint

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="predict_", delete=False,
    )
    json.dump(params, tmp, indent=2)
    tmp.close()

    extra_env = {"COSMOS_PREDICT_REPO": repo_dir} if repo_dir else None
    proc = just_run("predict-generate", tmp.name, timeout_s=60 * 60 * 3,
                    extra_env=extra_env)
    return proc_result(
        proc,
        success_text=f"✅ Predict2.5 ({model_variant}) → {output_dir}",
        fail_text=f"predict generate failed: {proc.get('stderr', '')[:300]}",
    )
