# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos 3 extra tools — prompt upsampling, batch captioning, VideoPhy2 eval.

Thin wrappers over justfile recipes (c3-upsample / c3-caption-batch /
c3-eval-videophy2), themselves backed by cosmos_framework scripts. These close
the upstream parity gaps for generator prompt-upsampling and task-specific
evaluation.

  - cosmos3_upsample_prompt : short scene desc -> dense structured prompt
  - cosmos3_caption_batch   : batch video captioning (VLM, reasoner-backed)
  - cosmos3_eval_videophy2  : VideoPhy-2 physical-plausibility benchmark
"""
from __future__ import annotations

from strands import tool

from ._security import (
    SecurityError,
    resolve_in_workspace,
    resolve_output_path,
)

from ._common import just_run, proc_result

_UPSAMPLE_TIMEOUT = 60 * 10     # 10m: single LLM call (max_tokens=20000)
_CAPTION_TIMEOUT = 60 * 60      # 1h: batch over a directory of videos
_EVAL_TIMEOUT = 60 * 60 * 4     # 4h: run+eval over a val manifest


@tool
def cosmos3_upsample_prompt(
    description: str,
    task: str = "t2v",
    port: int = 8000,
    aspect: str = "16,9",
    width: int = 832,
    height: int = 480,
    fps: int = 24,
    duration: int = 8,
    image: str = "",
) -> dict:
    """Cosmos 3 prompt upsampling: expand a short scene description into a dense,
    structured generator prompt (the recommended Generator input path).

    Uses the canonical v4.2 upsampler template + Cosmos 3 sampling defaults
    (max_tokens=20000, temperature=0.7, top_p=0.8, top_k=20, presence=1.5,
    seed=3407) and queries a running reasoner server (`just c3-serve-reason`).

    Args:
        description: Short source scene description to upsample.
        task: Generation task the prompt is for — "t2v" | "t2i" | "i2v".
        port: Reasoner vLLM server port (default 8000).
        aspect: Aspect ratio in comma form (e.g. "16,9", "1,1", "9,16").
        width: Output frame width in pixels (480p default: 832).
        height: Output frame height in pixels (480p default: 480).
        fps: Target FPS (t2v/i2v only).
        duration: Clip duration in whole seconds (t2v/i2v only).
        image: Conditioning image path/URL for i2v (optional).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    if task not in ("t2v", "t2i", "i2v"):
        from ._common import err
        return err("task must be one of: t2v, t2i, i2v")
    # `aspect` is still positionally interpolated into the recipe ({{aspect}});
    # constrain it to the N,N format so it can carry no shell/template
    # metacharacters (CWE-78 defense for the one remaining positional free-text).
    import re as _re
    if not _re.fullmatch(r"\d{1,3},\d{1,3}", str(aspect)):
        from ._common import err
        return err("aspect must look like 'W,H' (e.g. '16,9')")
    # description/image are LLM-controlled free-text/paths -> pass via env vars
    # (C3_UP_DESC/C3_UP_IMAGE), never as positional {{param}} interpolation (CWE-78).
    proc = just_run(
        "c3-upsample", task, str(port), aspect, str(width),
        str(height), str(fps), str(duration),
        timeout_s=_UPSAMPLE_TIMEOUT,
        extra_env={"C3_UP_DESC": str(description), "C3_UP_IMAGE": str(image or "")},
    )
    return proc_result(proc, "cosmos3 upsampled prompt:", "c3-upsample failed")


@tool
def cosmos3_caption_batch(
    video: str,
    out: str = "/tmp/c3_captions",
    port: int = 8000,
    workers: int = 16,
    template: str = "",
) -> dict:
    """Cosmos 3 batch video captioning via the framework VLM script.

    Captions a single video or every video in a directory, writing one caption
    file per input under `out`. Useful for SFT dataset preparation. Needs a
    running reasoner server (`just c3-serve-reason`).

    Args:
        video: Path to a single video file OR a directory of videos.
        out: Output directory for generated captions.
        port: Reasoner vLLM server port (default 8000).
        workers: Max concurrent requests to the server.
        template: Optional custom prompt-template path. Empty => auto-resolve the
            built-in video_captioner.txt (handles the upstream default-path bug).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    # Video/out/template are LLM-controlled paths -> confine to the
    # workspace and pass via env (no {{param}} interpolation, CWE-78/CWE-22).
    from ._common import err
    try:
        video_p = str(resolve_in_workspace(video, must_exist=True))
        out_p = str(resolve_output_path(out))
        tmpl_p = str(resolve_in_workspace(template, must_exist=True)) if template else ""
    except SecurityError as e:
        return err(str(e))
    proc = just_run(
        "c3-caption-batch", str(int(port)), str(int(workers)),
        timeout_s=_CAPTION_TIMEOUT,
        extra_env={
            "C3_CAP_VIDEO": video_p,
            "C3_CAP_OUT": out_p,
            "C3_CAP_TEMPLATE": tmpl_p,
        },
    )
    return proc_result(proc, "cosmos3 batch captions -> " + out_p, "c3-caption-batch failed")


@tool
def cosmos3_eval_videophy2(
    results_dir: str,
    hf_ckpt: str = "",
    val_root: str = "",
    batch_size: int = 1,
    max_new_tokens: int = 256,
    nproc: int = 1,
) -> dict:
    """Cosmos 3 VideoPhy-2 evaluation (task-specific physical-plausibility benchmark).

    Two modes (mirrors upstream eval_videophy2):
      - run+eval : provide `hf_ckpt` + `val_root` — loads the HF safetensors
        export, runs batched generation over the val manifest, writes per-sample
        JSON + summary.json into `results_dir`.
      - eval-only: provide only `results_dir` (already filled by a prior run) —
        re-scores and rewrites summary.json.

    Multi-GPU data-parallel via torchrun when `nproc` > 1.

    Args:
        results_dir: Output/scan directory for per-sample JSON + summary.json.
        hf_ckpt: HF safetensors checkpoint dir (run+eval mode).
        val_root: Prepared VideoPhy-2 val manifest root (run+eval mode).
        batch_size: Per-rank generation batch size.
        max_new_tokens: Max new tokens per generation.
        nproc: GPUs for torchrun (1 = single-process on cuda:0).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    # Results_dir/hf_ckpt/val_root are LLM-controlled paths -> confine
    # to the workspace and pass via env (no {{param}} interpolation, CWE-78/CWE-22).
    from ._common import err
    try:
        results_p = str(resolve_output_path(results_dir))
        hf_p = str(resolve_in_workspace(hf_ckpt, must_exist=True)) if hf_ckpt else ""
        val_p = str(resolve_in_workspace(val_root, must_exist=True)) if val_root else ""
    except SecurityError as e:
        return err(str(e))
    proc = just_run(
        "c3-eval-videophy2",
        str(int(batch_size)), str(int(max_new_tokens)), str(int(nproc)),
        timeout_s=_EVAL_TIMEOUT,
        extra_env={
            "C3_EVAL_RESULTS": results_p,
            "C3_EVAL_HF_CKPT": hf_p,
            "C3_EVAL_VAL_ROOT": val_p,
        },
    )
    return proc_result(
        proc, "cosmos3 videophy2 eval -> " + results_p + "/summary.json",
        "c3-eval-videophy2 failed",
    )
