# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper around `just rtp-capture` — GStreamer RTP/H.264 frame capture."""
from __future__ import annotations

import tempfile
from pathlib import Path

from strands import tool
from ._common import just_run, ok, err
from ._security import SecurityError, resolve_output_path


@tool
def rtp_capture_frame(
    bind_ip: str = "0.0.0.0",
    port: int = 5600,
    width: int = 800,
    height: int = 600,
    timeout_s: int = 5,
    output_path: str = "",
    return_image: bool = True,
) -> dict:
    """Capture one JPEG from an RTP/H.264 stream via `just rtp-capture`.

    The recipe tries Jetson HW decode (nvv4l2decoder/nvjpegenc),
    falling back to software decode automatically.

    Args:
        bind_ip: UDP bind IP.
        port: RTP UDP port.
        width / height: expected frame size.
        timeout_s: give up after N seconds.
        output_path: where to save JPEG (default: temp file).
        return_image: if True, embed JPEG bytes in the response.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the captured frame's path, with the image embedded in the result; on error ``status`` is ``"error"`` with a message.
    """
    if not output_path:
        output_path = tempfile.mktemp(suffix=".jpg", prefix="cosmos_rtp_")

    # Output_path is LLM-controlled -> confine to workspace and pass via
    # $RTP_OUTPUT env (no {{param}} interpolation, CWE-78/CWE-22). bind_ip likewise
    # flows through $RTP_BIND. Numerics stay positional (validated as ints).
    try:
        output_path = str(resolve_output_path(output_path))
    except SecurityError as e:
        return err(str(e))

    proc = just_run(
        "rtp-capture",
        str(int(port)), str(int(width)), str(int(height)), str(int(timeout_s)),
        timeout_s=timeout_s + 10,
        extra_env={"RTP_BIND": str(bind_ip), "RTP_OUTPUT": output_path},
    )

    p = Path(output_path)
    captured = p.exists() and p.stat().st_size > 0

    if not captured:
        return err(
            "no frame captured",
            data={"stderr": proc.get("stderr", "")[-600:], "cmd": proc.get("cmd")},
        )

    image_bytes = p.read_bytes() if return_image else None
    return ok(
        text=f"📸 captured {p.stat().st_size} bytes → {output_path}",
        data={"image_path": str(p), "size": p.stat().st_size,
              "width": width, "height": height},
        image_bytes=image_bytes,
        image_format="jpeg",
    )
