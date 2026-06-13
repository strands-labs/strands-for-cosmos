# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video utilities: probe metadata and extract frames via ffprobe/ffmpeg.

``video_probe`` returns resolution / fps / duration / codec; ``video_extract_frames``
samples frames to JPEGs. Both read and write inside the project workspace and call
ffprobe/ffmpeg directly (no shell), so paths are handled safely.
"""
from __future__ import annotations

import json
import tempfile

from strands import tool

from ._common import err, ok
from ._security import (
    SecurityError,
    resolve_in_workspace,
    resolve_output_path,
    safe_run,
)


@tool
def video_probe(video_path: str) -> dict:
    """Get video metadata via ffprobe (JSON).

    Args:
        video_path: Path to video file (must be inside the workspace).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the video's metadata (resolution, fps, duration, codec, frame count) as both a human summary and a ``json`` block; on error ``status`` is ``"error"`` with a message.
    """
    try:
        p = resolve_in_workspace(video_path, must_exist=True)
    except SecurityError as e:
        return err(str(e))

    proc = safe_run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(p),
        ],
        timeout_s=30,
    )
    if not proc.get("ok"):
        return err(f"ffprobe failed: {proc.get('stderr', '')[:200]}")

    try:
        raw = proc.get("stdout", "")
        json_start = raw.find("{")
        data = json.loads(raw[json_start:]) if json_start >= 0 else {}
        vstream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        rate = vstream.get("r_frame_rate", "0/1")
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            fps = 0.0
        summary = {
            "duration": float(data.get("format", {}).get("duration", 0) or 0),
            "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
            "codec": vstream.get("codec_name"),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": round(fps, 2),
            "pix_fmt": vstream.get("pix_fmt"),
            "nb_frames": vstream.get("nb_frames"),
        }
        return ok(
            f"\U0001F4F9 {p.name}: {summary['width']}x{summary['height']} @ "
            f"{summary['fps']}fps, {summary['duration']:.1f}s, {summary['codec']}",
            data={"summary": summary},
        )
    except Exception as e:
        return err(f"probe parse failed: {e}")


@tool
def video_extract_frames(
    video_path: str,
    output_dir: str = "",
    fps: float = 1.0,
    max_frames: int = 0,
    return_first: bool = True,
) -> dict:
    """Extract frames from a video via ffmpeg.

    Args:
        video_path: Path to input video (inside the workspace).
        output_dir: Output dir inside the workspace (default: temp).
        fps: Frames/sec to extract (1.0 = every second).
        max_frames: Stop after N frames (0 = unlimited).
        return_first: Embed the first frame in the response.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the output directory, frame count, and the first/last frame paths, with the first frame embedded as an image; on error ``status`` is ``"error"`` with a message.
    """
    try:
        p = resolve_in_workspace(video_path, must_exist=True)
    except SecurityError as e:
        return err(str(e))

    # Validate numeric args (defense in depth -- they go into argv, not a shell).
    try:
        fps_val = float(fps)
        if fps_val <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return err("fps must be a positive number")
    try:
        max_frames_val = int(max_frames)
        if max_frames_val < 0:
            raise ValueError
    except (TypeError, ValueError):
        return err("max_frames must be a non-negative integer")

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="cosmos_frames_")
    try:
        outp = resolve_output_path(output_dir)
    except SecurityError as e:
        return err(str(e))
    outp.mkdir(parents=True, exist_ok=True)

    argv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(p), "-vf", f"fps={fps_val}",
    ]
    if max_frames_val != 0:
        argv += ["-frames:v", str(max_frames_val)]
    argv.append(str(outp / "frame_%06d.jpg"))

    proc = safe_run(argv, timeout_s=60 * 30)
    # ffmpeg may exit non-zero on partial input; check for produced frames.
    frames = sorted(outp.glob("frame_*.jpg"))
    if not frames:
        return err(
            f"no frames extracted: {proc.get('stderr', '')[:200]}",
            data={"output_dir": str(outp)},
        )

    first_bytes = frames[0].read_bytes() if return_first else None
    return ok(
        text=f"\U0001F4FC extracted {len(frames)} frame(s) -> {outp}",
        data={
            "output_dir": str(outp),
            "frame_count": len(frames),
            "first_frame": str(frames[0]),
            "last_frame": str(frames[-1]),
        },
        image_bytes=first_bytes,
        image_format="jpeg",
    )
