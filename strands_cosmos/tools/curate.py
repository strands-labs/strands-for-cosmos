# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just curate` (Cosmos-Xenna data pipeline)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result
from ._security import SecurityError, validate_identifier


@tool
def cosmos_curate(
    input_dir: str,
    output_dir: str = "./outputs/curated",
    stages: str = "all",
    num_workers: int = 8,
    repo_dir: str = "",
) -> dict:
    """Run the Cosmos-Xenna data curation pipeline via `just curate`.

    Stages (comma-separated): split, transcode, crop, filter, caption, dedup, shard.
    Use "all" for every stage.

    Args:
        input_dir: Directory of raw videos.
        output_dir: Curated output destination.
        stages: "all" or comma-separated stage names.
        num_workers: Ray workers.
        repo_dir: Override COSMOS_XENNA_REPO.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the curated dataset output; on error ``status`` is ``"error"`` with a message.
    """
    try:
        input_dir = validate_identifier(input_dir, what="input_dir")
        output_dir = validate_identifier(output_dir, what="output_dir")
        stages = validate_identifier(stages, what="stages")
    except SecurityError as e:
        from ._common import err
        return err(str(e))
    extra_env = {"COSMOS_XENNA_REPO": repo_dir} if repo_dir else None
    proc = just_run(
        "curate", input_dir, output_dir, stages, str(num_workers),
        timeout_s=60 * 60 * 24, extra_env=extra_env,
    )
    return proc_result(
        proc,
        success_text=f"✅ curation ({stages}) → {output_dir}",
        fail_text=f"curation failed: {proc.get('stderr', '')[:300]}",
    )
