# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an image from disk and embed it in the tool result so the agent can see it.

Reads an image file from inside the project workspace and returns it as visual
content the model can reason over. Supports PNG, JPEG, GIF, and WebP.
"""
from __future__ import annotations

from strands import tool

from ._common import err, ok
from ._security import SecurityError, resolve_in_workspace


@tool
def image_read(image_path: str) -> dict:
    """Read an image file and embed it in the response so the agent can see it.

    Supports PNG, JPEG/JPG, GIF, WebP. The path must be inside the workspace.

    Args:
        image_path: Path to image on disk (inside the workspace).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the requested image embedded as visual content the model can see; on error ``status`` is ``"error"`` with a message.
    """
    try:
        p = resolve_in_workspace(image_path, must_exist=True)
    except SecurityError as e:
        return err(str(e))

    fmt = p.suffix.lower().lstrip(".") or "png"
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "gif", "webp"):
        return err(f"unsupported image format: {p.suffix!r} (allowed: png/jpg/jpeg/gif/webp)")

    data = p.read_bytes()
    return ok(
        text=f"\U0001F4F7 loaded {p.name} ({len(data)} bytes, {fmt})",
        data={"path": str(p), "size": len(data), "format": fmt},
        image_bytes=data,
        image_format=fmt,
    )
