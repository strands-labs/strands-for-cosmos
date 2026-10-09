# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Edge tools that need neither a vLLM server nor the Cosmos Framework.

The Edge checkpoint loads with released ``diffusers>=0.40`` (``Cosmos3OmniPipeline``
+ ``CosmosActionCondition``), so robot-policy / forward-dynamics rollouts can run
in-process on a single edge GPU. MEASURED on a Jetson AGX Thor (diffusers 0.41.0,
torch 2.14.1+cu130, Edge rev 344d602b, 8 steps, 480p tier):
policy droid_lerobot chunk 16 -> actions (16,10) + 17 frames in 60.2 s (first call)
/ 35.6 s (warm, bridge_orig_lerobot); forward_dynamics 36.1 s; peak 12.09 GiB.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from strands import tool

from ..cosmos3_models import EDGE, require_surface, resolve_model_id, validate_embodiment
from ._common import err, ok
from ._security import SecurityError, resolve_in_workspace, resolve_output_path

logger = logging.getLogger(__name__)

_pipelines: Dict[str, Any] = {}
_MODES = ("policy", "forward_dynamics", "inverse_dynamics")
_VIEWS = ("ego_view", "third_person_view", "wrist_view", "concat_view")


def _get_pipeline(model_id: str):
    """One ``Cosmos3OmniPipeline`` per checkpoint (7.6 GiB on GPU for Edge; load once)."""
    pipe = _pipelines.get(model_id)
    if pipe is None:
        import torch
        from diffusers import Cosmos3OmniPipeline

        pipe = Cosmos3OmniPipeline.from_pretrained(model_id, dtype=torch.bfloat16, safety_checker=None)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        _pipelines[model_id] = pipe
    return pipe


def _load_frame(path: str):
    from PIL import Image

    p = resolve_in_workspace(path, must_exist=True)
    if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
        import av

        with av.open(str(p)) as c:
            for f in c.decode(video=0):
                return f.to_image()
        raise ValueError(f"no decodable frame in {p}")
    return Image.open(p).convert("RGB")


@tool
def cosmos3_action_edge(
    image: str,
    prompt: str = "",
    mode: str = "policy",
    embodiment: str = "droid_lerobot",
    chunk_size: int = 16,
    steps: int = 8,
    guidance: float = 7.0,
    view_point: str = "third_person_view",
    raw_actions_json: str = "",
    out: str = "/tmp/c3_edge_action",
    seed: int = 0,
    model: str = EDGE,
    allow_untrained: bool = False,
) -> dict:
    """Cosmos3-Edge world-model action rollout in-process (Diffusers, no server, no Cosmos Framework).

    Args:
        image: Conditioning frame (image path, or a video whose first frame is used). Workspace-confined.
        prompt: Task instruction (e.g. "pick up the blue cube").
        mode: "policy" (predict actions + future video), "forward_dynamics" (video from raw_actions_json),
            or "inverse_dynamics" (actions connecting the frames of a video given as `image`).
        embodiment: Trained Cosmos3-Edge embodiment: av, camera_pose, hand_pose, umi, bridge_orig_lerobot,
            droid_lerobot, robomind-franka, robomind-franka-dual, robomind-ur, agibotworld, fractal.
            so101-class arms are not registered; pusht/libero/galbot have untrained heads and are refused.
        chunk_size: Action transition steps (video spans chunk_size + 1 frames).
        steps: Diffusion steps (8 is a fast preview; the card uses 35).
        guidance: Classifier-free guidance scale.
        view_point: ego_view | third_person_view | wrist_view | concat_view.
        raw_actions_json: JSON list [T][raw_action_dim] driving forward_dynamics.
        out: Output directory for actions.json + frame_000.png ... (workspace-confined).
        seed: Random seed.
        model: Checkpoint id/path (default nvidia/Cosmos3-Edge).
        allow_untrained: Run a registered-but-untrained embodiment anyway (outputs will be noise).

    Returns:
        A Strands tool-result dict ``{"status", "content"}`` with the action matrix
        (chunk_size x raw_action_dim), frame count, seconds and output paths.
    """
    if mode not in _MODES:
        return err(f"mode must be one of {_MODES}, got {mode!r}")
    if view_point not in _VIEWS:
        return err(f"view_point must be one of {_VIEWS}, got {view_point!r}")
    try:
        model_id = require_surface(resolve_model_id(model), "action")
        emb = validate_embodiment(embodiment, allow_untrained=allow_untrained)
        frame = _load_frame(image)
        out_dir = resolve_output_path(out)
    except (ValueError, SecurityError, FileNotFoundError) as e:
        return err(str(e))

    import time

    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

    raw: Optional[torch.Tensor] = None
    if mode == "forward_dynamics":
        if not raw_actions_json:
            return err("forward_dynamics needs raw_actions_json: a JSON list of [T][raw_action_dim] actions")
        try:
            raw = torch.tensor(json.loads(raw_actions_json), dtype=torch.float32)
        except (ValueError, TypeError) as e:
            return err(f"raw_actions_json is not a numeric 2-D JSON list: {e}")
        if raw.ndim != 2 or (emb.raw_action_dim and raw.shape[1] != emb.raw_action_dim):
            return err(f"raw_actions_json must be [T][{emb.raw_action_dim}] for {emb.name}, got {list(raw.shape)}")

    cond_kw: Dict[str, Any] = dict(mode=mode, chunk_size=int(chunk_size), domain_name=emb.name, view_point=view_point)
    if mode == "inverse_dynamics":
        cond_kw["video"] = [frame] * (int(chunk_size) + 1)
    else:
        cond_kw["image"] = frame
    if raw is not None:
        cond_kw["raw_actions"] = raw

    try:
        pipe = _get_pipeline(model_id)
        gen = torch.Generator(pipe.device).manual_seed(int(seed))
        t0 = time.time()
        result = pipe(prompt=prompt or "", action=CosmosActionCondition(**cond_kw),
                      num_inference_steps=int(steps), guidance_scale=float(guidance), generator=gen)
        secs = round(time.time() - t0, 2)
    except Exception as e:  # pragma: no cover - GPU path
        return err(f"cosmos3-edge action failed: {e}")

    actions: Optional[List[List[float]]] = None
    act = result.action
    if act is not None:
        t = act[0] if isinstance(act, (list, tuple)) else act
        actions = [[round(float(x), 5) for x in row] for row in t.float().cpu().tolist()]
    frames = result.video or []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(out_dir / f"frame_{i:03d}.png")
    (out_dir / "actions.json").write_text(json.dumps({"embodiment": emb.name, "domain_id": emb.domain_id,
                                                      "mode": mode, "actions": actions}))
    data = {"model": model_id, "embodiment": emb.name, "domain_id": emb.domain_id,
            "raw_action_dim": emb.raw_action_dim, "mode": mode, "seconds": secs,
            "n_frames": len(frames), "action_shape": [len(actions), len(actions[0])] if actions else None,
            "actions": actions, "out": str(out_dir)}
    shape = data["action_shape"]
    return ok(text=f"cosmos3-edge {mode} [{emb.name}] -> actions {shape}, {len(frames)} frames in {secs}s -> {out_dir}",
              data=data)


__all__ = ["cosmos3_action_edge"]
