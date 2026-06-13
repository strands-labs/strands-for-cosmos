# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just transfer-generate` (Cosmos-Transfer 2.5)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from strands import tool
from ._security import SecurityError, validate_identifier
from ._common import just_run, proc_result, err


@tool
def cosmos_transfer_generate(
    prompt: str,
    control: str = "edge",
    output_dir: str = "./outputs/transfer2_5",
    control_video: str = "",
    style_image: str = "",
    control_weights: str = "",
    guidance_scale: float = 3.0,
    num_steps: int = 35,
    seed: int = 0,
    repo_dir: str = "",
) -> dict:
    """Generate video with Cosmos-Transfer2.5 (ControlNet style/structure).

    Control types: edge | depth | seg | vis | multi.

    Args:
        prompt: Scene description.
        control: edge | depth | seg | vis | multi.
        output_dir: Output directory for .mp4.
        control_video: Pre-computed control video path (optional).
        style_image: Style reference image (for image-prompt feature).
        control_weights: JSON string of weights for control='multi' (e.g. '{"edge":0.5,"depth":0.5}').
        guidance_scale / num_steps / seed: Diffusion params.
        repo_dir: Override COSMOS_TRANSFER_REPO.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the generated (transformed) video; on error ``status`` is ``"error"`` with a message.
    """
    if control not in {"edge", "depth", "seg", "vis", "multi"}:
        return err(f"control must be one of edge/depth/seg/vis/multi, got {control!r}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params: dict = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps,
        "seed": seed,
        "output_dir": output_dir,
    }
    if control_video:
        params["control_path"] = control_video
    if style_image:
        params["style_image"] = style_image
    if control == "multi":
        if not control_weights:
            return err("control='multi' requires control_weights JSON string")
        try:
            params["control_weights"] = json.loads(control_weights)
        except json.JSONDecodeError as e:
            return err(f"invalid control_weights JSON: {e}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="transfer_", delete=False,
    )
    json.dump(params, tmp, indent=2)
    tmp.close()

    extra_env = {"COSMOS_TRANSFER_REPO": repo_dir} if repo_dir else None
    cli_control = "edge" if control == "multi" else control
    # cli_control is already constrained to the control enum above; assert the
    # charset explicitly so the no-interpolation gate can prove it (CWE-78).
    cli_control = validate_identifier(cli_control, what="control")
    proc = just_run("transfer-generate", tmp.name, cli_control,
                    timeout_s=60 * 60 * 3, extra_env=extra_env)
    return proc_result(
        proc,
        success_text=f"✅ Transfer2.5 ({control}) → {output_dir}",
        fail_text=f"transfer generate failed: {proc.get('stderr', '')[:300]}",
    )
