# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos vision inference as a Strands tool."""

import json
import logging
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

_cached_model = None


def _get_model(model_id: str = "nvidia/Cosmos-Reason2-2B"):
    global _cached_model
    if _cached_model is None or _cached_model.config["model_id"] != model_id:
        from strands_cosmos.cosmos_vision_model import CosmosVisionModel
        _cached_model = CosmosVisionModel(model_id=model_id)
    return _cached_model


@tool
def cosmos_vision_invoke(
    prompt: str,
    video_path: str = "",
    image_path: str = "",
    model_id: str = "nvidia/Cosmos-Reason2-2B",
    reasoning: bool = False,
    task: str = "",
    fps: float = 4.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Invoke NVIDIA Cosmos Reason for video/image understanding.

    Specialized for physical AI: driving analysis, robot planning,
    video captioning, temporal reasoning, and embodied intelligence.

    Args:
        prompt: Text prompt for the model.
        video_path: Path to video file.
        image_path: Path to image file.
        model_id: Cosmos model ID.
        reasoning: Enable chain-of-thought reasoning.
        task: Predefined task (caption, driving, embodied_reasoning, causal, robot_cot, etc.)
        fps: Video frame rate for processing.
        max_tokens: Maximum output tokens.

    Returns:
        Dict with status and generated text.
    """
    try:
        import asyncio
        from strands_cosmos.cosmos_vision_model import TASK_PROMPTS

        # Use task prompt if specified
        if task and task in TASK_PROMPTS:
            prompt = TASK_PROMPTS[task]

        # Build prompt with media tags
        full_prompt = ""
        if video_path:
            full_prompt += f"<video>{video_path}</video> "
        if image_path:
            full_prompt += f"<image>{image_path}</image> "
        full_prompt += prompt

        model = _get_model(model_id)
        model.config["reasoning"] = reasoning
        model.config["fps"] = fps
        model.config["params"] = {"max_tokens": max_tokens}

        messages = [{"role": "user", "content": [{"text": full_prompt}]}]
        response_text = ""

        async def _run():
            nonlocal response_text
            async for event in model.stream(messages):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        response_text += delta["text"]

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        return {
            "status": "success",
            "content": [{"text": response_text}],
        }
    except Exception as e:
        logger.error(f"cosmos_vision_invoke error: {e}")
        return {"status": "error", "content": [{"text": str(e)}]}
