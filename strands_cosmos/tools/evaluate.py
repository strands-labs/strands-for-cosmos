# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just evaluate` — all Cosmos metrics."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err
from ._security import SecurityError, validate_identifier


VALID_METRICS = {
    "fid", "fvd", "tse", "cse", "sampson",
    "blur_ssim", "canny_f1", "depth_rmse", "seg_miou", "dover",
    "reason_critic", "reason_reward",
}


@tool
def cosmos_evaluate(
    metric: str,
    pred_path: str,
    gt_path: str = "",
    output_dir: str = "./outputs/eval",
    repo_dir: str = "",
) -> dict:
    """Run a Cosmos evaluation metric via `just evaluate`.

    Metrics:
      Predict quality: fid, fvd, tse, cse, sampson
      Transfer/Control: blur_ssim, canny_f1, depth_rmse, seg_miou, dover
      VLM reasoning: reason_critic, reason_reward

    Args:
        metric: One of the valid metrics.
        pred_path: Predicted video/image path.
        gt_path: Ground-truth path (required for most).
        output_dir: JSON results destination.
        repo_dir: Override COSMOS_COOKBOOK_REPO.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the computed evaluation metrics; on error ``status`` is ``"error"`` with a message.
    """
    if metric not in VALID_METRICS:
        return err(f"unknown metric: {metric}", data={"known": sorted(VALID_METRICS)})

    try:
        pred_path = validate_identifier(pred_path, what="pred_path")
        gt_path = validate_identifier(gt_path, what="gt_path", allow_empty=True)
        output_dir = validate_identifier(output_dir, what="output_dir")
    except SecurityError as e:
        return err(str(e))
    extra_env = {"COSMOS_COOKBOOK_REPO": repo_dir} if repo_dir else None
    proc = just_run(
        "evaluate", metric, pred_path, gt_path, output_dir,
        timeout_s=60 * 60 * 2, extra_env=extra_env,
    )
    return proc_result(
        proc,
        success_text=f"✅ {metric} eval → {output_dir}",
        fail_text=f"{metric} eval failed: {proc.get('stderr', '')[:300]}",
    )
