# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Reason2 VLM inference against TRT-Edge-LLM HTTP server."""
from __future__ import annotations

import base64
import time

from strands import tool

from ._common import err, ok
from ._security import SecurityError, resolve_in_workspace, validate_url


@tool
def cosmos_inference(
    prompt: str,
    image_path: str = "",
    image_b64: str = "",
    server_url: str = "",
    max_tokens: int = 256,
    temperature: float = 0.2,
    system_prompt: str = "",
    return_image: bool = False,
) -> dict:
    """Run Cosmos-Reason2 VLM inference via TRT-Edge-LLM HTTP server.

    Uses the local quantized server (FP8 on Jetson Thor). For HF-based
    full-precision inference, use `cosmos_reason_hf` instead.

    Args:
        prompt: User instruction (e.g. "count people, report clothing").
        image_path: Path to JPEG/PNG on disk. Mutually exclusive with image_b64.
        image_b64: Base64 image (alternative to image_path).
        server_url: Override the VLM endpoint. Default: env COSMOS_VLM_URL or http://127.0.0.1:8080/v1/chat/completions.
        max_tokens: Token cap (keep low for Thor latency).
        temperature: Sampling temperature (0.0-1.0).
        system_prompt: Optional system prompt.
        return_image: If True, include the input image in the response.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text response; on error ``status`` is ``"error"`` with a message.
    """
    import os

    try:
        import requests
    except ImportError:
        return err("requests not installed: pip install requests")

    url = server_url or os.getenv("COSMOS_VLM_URL", "http://127.0.0.1:8080/v1/chat/completions")

    # SSRF guard: validate the (possibly LLM-supplied) endpoint against the
    # host allow-list and block private/link-local/metadata targets (CWE-918).
    try:
        url = validate_url(url)
    except SecurityError as e:
        return err(str(e), data={"url": url})

    if image_path and image_b64:
        return err("provide exactly one of image_path or image_b64")
    if not image_path and not image_b64:
        return err("image_path or image_b64 is required")

    image_bytes: bytes | None = None
    if image_path:
        # Confine the (LLM-supplied) image path to the workspace (CWE-22).
        try:
            p = resolve_in_workspace(image_path, must_exist=True)
        except SecurityError as e:
            return err(str(e))
        image_bytes = p.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    })
    payload = {
        "model": "trt-edgellm",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=60)
        latency_ms = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            text = str(data)

        result_data = {
            "latency_ms": latency_ms,
            "server_url": url,
            "prompt_chars": len(prompt),
            "output_chars": len(text),
        }

        return ok(
            text=f"VLM → {text}\n\n(latency: {latency_ms}ms)",
            data=result_data,
            image_bytes=image_bytes if (return_image and image_bytes) else None,
            image_format="jpeg",
        )
    except Exception as e:
        if "ConnectionError" in type(e).__name__ or "connection" in str(e).lower():
            return err(
                f"cannot reach VLM server at {url}. Start with cosmos_serve(action='start', ...)",
                data={"url": url},
            )
        return err(f"{type(e).__name__}: {e}", data={"url": url})
