#!/usr/bin/env python3
"""Inspect and stage media before it reaches a Cosmos model - runs anywhere.

Goal:
    Show the four Cosmos tools that need no GPU and no model download. They are
    the first mile of every vision pipeline: check the host, probe a video, pull
    frames, and read an image. Use them to validate inputs before you spend GPU
    time on a reasoner or generator.

What it shows:
    cosmos_sysinfo          - host / GPU summary (is a model even runnable here?)
    video_probe             - resolution, fps, duration, codec of a clip
    video_extract_frames    - sample frames from a clip to a directory
    image_read              - load an image into a tool result (base64 block)

Dependencies:
    uv pip install strands-cosmos      # core install, no extra needed
    ffmpeg / ffprobe on PATH           # video_probe + video_extract_frames use them

Expected output:
    Four sections, each ending in "ok". A frames/ directory with sampled JPGs.
    No "PASS" is printed unless every step actually returned status="success".

Runtime:
    Under 5 seconds on a laptop. No GPU, no network, no model weights.

Run:
    python examples/11_media_prep.py
    SAMPLE_VIDEO=/path/clip.mp4 SAMPLE_IMAGE=/path/frame.png python examples/11_media_prep.py

Learn-first notebook: ../notebooks/05_tool_usage.ipynb
    ("Cosmos as Tools" - the same no-GPU tools, explained step by step.)
"""

import os
import sys

from strands_cosmos import (
    cosmos_sysinfo,
    image_read,
    video_extract_frames,
    video_probe,
)

# Cosmos tools confine file reads/writes to a workspace allow-list, read at call
# time. Our sample media sits in the project root and we write frames to the
# system tempdir, so widen the workspace to both before the first tool call.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COSMOS_WORKSPACE", os.pathsep.join([PROJECT_ROOT, "/tmp"]))

SAMPLE_VIDEO = os.environ.get("SAMPLE_VIDEO", os.path.join(PROJECT_ROOT, "sample.mp4"))
SAMPLE_IMAGE = os.environ.get("SAMPLE_IMAGE", os.path.join(PROJECT_ROOT, "sample.png"))
FRAMES_DIR = os.path.join("/tmp", "cosmos_media_prep_frames")


def _check(result: dict, label: str) -> dict:
    """Print a tool result and stop the run if it did not succeed.

    Surfaces the real error instead of pretending the step worked - the same
    honesty a caller downstream would need before feeding this media to a model.
    """
    status = result.get("status")
    text = result["content"][0]["text"] if result.get("content") else ""
    print(text.rstrip())
    if status != "success":
        print(f"[{label}] failed with status={status!r}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[{label}] ok\n")
    return result


def main() -> int:
    print("=== 11: Media prep (no GPU required) ===\n")

    print("--- 1. Host summary ---")
    _check(cosmos_sysinfo(), "sysinfo")

    if not os.path.exists(SAMPLE_VIDEO):
        print(f"sample video not found: {SAMPLE_VIDEO}", file=sys.stderr)
        print("Set SAMPLE_VIDEO to a real clip and re-run.", file=sys.stderr)
        return 1

    print("--- 2. Probe the video ---")
    _check(video_probe(video_path=SAMPLE_VIDEO), "video_probe")

    print("--- 3. Extract frames ---")
    _check(
        video_extract_frames(
            video_path=SAMPLE_VIDEO,
            output_dir=FRAMES_DIR,
            fps=1.0,          # one frame per second; raise for a denser sample
            max_frames=8,     # cap the count so a long clip stays cheap
        ),
        "video_extract_frames",
    )
    frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    print(f"frames on disk: {len(frames)} -> {FRAMES_DIR}")
    print(f"first few: {frames[:3]}\n")

    if not os.path.exists(SAMPLE_IMAGE):
        print(f"sample image not found: {SAMPLE_IMAGE} (skipping image_read)", file=sys.stderr)
    else:
        print("--- 4. Read an image into a tool result ---")
        result = _check(image_read(image_path=SAMPLE_IMAGE), "image_read")
        kinds = [next(iter(block)) for block in result["content"]]
        print(f"content blocks: {kinds}\n")

    print("=== PASS: media inspected and staged, ready for a Cosmos model ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
