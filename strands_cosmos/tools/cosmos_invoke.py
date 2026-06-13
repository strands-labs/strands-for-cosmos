# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos text inference as a Strands tool."""

import json
import logging
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

_cached_model = None


def _get_model(model_id: str = "nvidia/Cosmos-Reason2-2B"):
    global _cached_model
    if _cached_model is None or _cached_model.config["model_id"] != model_id:
        from strands_cosmos.cosmos_model import CosmosModel
        _cached_model = CosmosModel(model_id=model_id)
    return _cached_model


@tool
def cosmos_invoke(
    prompt: str,
    model_id: str = "nvidia/Cosmos-Reason2-2B",
    reasoning: bool = False,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Invoke NVIDIA Cosmos Reason model for physical AI reasoning.

    Args:
        prompt: Text prompt for the model.
        model_id: Cosmos model ID.
        reasoning: Enable chain-of-thought reasoning.
        max_tokens: Maximum output tokens.

    Returns:
        Dict with status and generated text.
    """
    try:
        import asyncio

        model = _get_model(model_id)
        model.config["reasoning"] = reasoning
        model.config["params"] = {"max_tokens": max_tokens}

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        response_text = ""

        async def _run():
            nonlocal response_text
            async for event in model.stream(messages):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        response_text += delta["text"]

        asyncio.get_event_loop().run_until_complete(_run())

        return {
            "status": "success",
            "content": [{"text": response_text}],
        }
    except Exception as e:
        logger.error(f"cosmos_invoke error: {e}")
        return {"status": "error", "content": [{"text": str(e)}]}
