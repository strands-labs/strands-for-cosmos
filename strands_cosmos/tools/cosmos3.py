# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos 3 tools — thin wrappers over justfile `c3-*` recipes.

Reasoner (text out, via local vLLM server):
    cosmos3_reason, cosmos3_caption, cosmos3_temporal, cosmos3_embodied,
    cosmos3_ground, cosmos3_plausibility, cosmos3_situation, cosmos3_action_cot
Generator (media out, via in-proc Diffusers):
    cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
    cosmos3_text2video_sound
Action (world-model, via Cosmos Framework torchrun):
    cosmos3_forward_dynamics, cosmos3_inverse_dynamics, cosmos3_policy
Server lifecycle:
    cosmos3_serve

All recipes live in the top-level justfile (single source of truth).
"""
from __future__ import annotations

from strands import tool

from ._common import err, just_run, ok, proc_result
from ._security import SecurityError, resolve_in_workspace, resolve_output_path

# Long timeouts: video gen + model load can take many minutes.
_GEN_TIMEOUT = 60 * 60        # 1h for generation
_REASON_TIMEOUT = 60 * 20     # 20m for reasoning (includes first-call warmup)
_ACTION_TIMEOUT = 60 * 60     # 1h for action rollouts


# Map a media path/URL extension -> Strands media `format` literal.
_IMAGE_FMTS = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "gif": "gif", "webp": "webp"}
_VIDEO_FMTS = {"mp4": "mp4", "mov": "mov", "mkv": "mkv", "webm": "webm", "flv": "flv",
               "mpeg": "mpeg", "mpg": "mpg", "wmv": "wmv", "3gp": "three_gp"}


def _media_format(path_or_url: str, default: str) -> str:
    """Infer a Strands media `format` from a file extension (defensive fallback)."""
    import os
    ext = os.path.splitext(str(path_or_url).split("?")[0])[1].lstrip(".").lower()
    return _IMAGE_FMTS.get(ext) or _VIDEO_FMTS.get(ext) or default


# Cache one reasoner provider per base_url so we reuse the OpenAI client.
_reasoner_models: dict = {}


def _get_reasoner(port: int):
    """Return a cached ``Cosmos3ReasonerModel`` bound to the local vLLM server.

    Using the SDK model provider (rather than shelling out to ``just c3-reason``)
    gives one consistent media path: the provider confines local files to the
    project workspace and base64-encodes them, and remote URLs go through the
    provider's URL policy. The tool therefore behaves exactly like a Strands
    ``Agent(model=Cosmos3ReasonerModel())`` would.
    """
    base_url = f"http://localhost:{int(port)}/v1"
    model = _reasoner_models.get(base_url)
    if model is None:
        from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel
        model = Cosmos3ReasonerModel(base_url=base_url)
        _reasoner_models[base_url] = model
    return model


def _reason(prompt, image, video, task, port, max_tokens, think):
    """Run Cosmos 3 reasoning via the SDK `Cosmos3ReasonerModel` provider.

    prompt/image/video are LLM-controlled; image/video are emitted as
    `<image>`/`<video>` tags so the provider's hardened `_media_to_url`
    validates + confines them (workspace allow-list / SSRF policy). Returns a
    proc-shaped dict ({returncode, stdout, stderr}) so existing `proc_result`
    callers keep working unchanged.
    """
    import asyncio
    import os

    # Build a single user turn from NATIVE ContentBlocks. We pass media as
    # {"image": {...}} / {"video": {...}} blocks (not <tag> strings) so the
    # provider routes each ref through its hardened `_media_to_url` resolver and
    # carries an explicit `format`. This is the correct Strands media shape and
    # mirrors how a real Agent would hand media to the model.
    text = str(prompt or "")
    if task:
        text = f"[task:{task}] {text}".strip()

    content: list = []
    if image:
        content.append({
            "image": {"source": {"url": str(image)}, "format": _media_format(image, "png")}
        })
    if video:
        content.append({
            "video": {"source": {"url": str(video)}, "format": _media_format(video, "mp4")}
        })
    if text:
        content.append({"text": text})
    messages = [{"role": "user", "content": content}]

    try:
        model = _get_reasoner(port)
        model.update_config(reasoning=bool(think), params={"max_tokens": int(max_tokens)})

        out_text = ""
        err_text = ""

        async def _run():
            nonlocal out_text, err_text
            async for event in model.stream(messages):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        chunk = delta["text"]
                        if chunk.startswith("[Cosmos3 error:"):
                            err_text += chunk
                        else:
                            out_text += chunk

        # Run the async provider stream to completion on a private loop.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        rc = 1 if err_text else 0
        return {"ok": rc == 0, "returncode": rc, "stdout": out_text,
                "stderr": err_text, "cmd": f"Cosmos3ReasonerModel@{model.config['base_url']}"}
    except Exception as e:  # SecurityError from media resolver, conn refused, etc.
        return {"ok": False, "returncode": 1, "stdout": "", "stderr": str(e),
                "cmd": "Cosmos3ReasonerModel"}


_GEN_MODES = {"text2image", "text2video", "image2video",
              "text2video-with-sound", "image2video-with-sound"}
_GEN_RES = {"480", "720", "1080"}


def _gen(mode, prompt, image, out, frames, fps, steps, guidance, res, sound, seed):
    """Invoke c3-gen with untrusted text/paths passed via env vars.

    The mode and resolution are constrained to fixed presets; the free-text
    prompt and the image/output paths are passed through environment variables
    so they are handled safely.
    """
    if mode not in _GEN_MODES:
        return err(f"invalid generation mode: {mode!r}")
    if str(res) not in _GEN_RES:
        return err(f"invalid resolution: {res!r} (allowed: {sorted(_GEN_RES)})")
    return just_run(
        "c3-gen", mode, str(frames), str(fps), str(steps), str(guidance), str(res),
        "true" if str(sound).lower() == "true" else "false", str(seed),
        timeout_s=_GEN_TIMEOUT,
        extra_env={
            "C3_GEN_PROMPT": str(prompt or ""),
            "C3_GEN_IMAGE": str(image or ""),
            "C3_GEN_OUT": str(out),
        },
    )


def _action(input_jsonl, out, checkpoint, seed):
    """Invoke the ``c3-action`` recipe with paths passed via environment variables.

    The input JSONL and output paths are confined to the project workspace; the
    checkpoint name flows through ``C3_ACTION_CKPT``. Only the integer seed and a
    fixed preset are passed positionally.
    """
    in_path = str(resolve_in_workspace(input_jsonl, must_exist=True))
    out_path = str(resolve_output_path(out))
    return just_run(
        "c3-action", str(int(seed)), "latency",
        timeout_s=_ACTION_TIMEOUT,
        extra_env={
            "C3_ACTION_INPUT": in_path,
            "C3_ACTION_OUT": out_path,
            "C3_ACTION_CKPT": str(checkpoint),
        },
    )


# ----- Reasoner -----------------------------------------------------------
@tool
def cosmos3_reason(
    prompt: str,
    image: str = "",
    video: str = "",
    task: str = "",
    port: int = 8000,
    max_tokens: int = 4096,
    think: bool = False,
) -> dict:
    """Cosmos 3 Reasoner: text+vision -> text via local vLLM server.

    Requires a running reasoner server (`just c3-serve-reason` / cosmos3_serve).

    Args:
        prompt: User instruction.
        image: Image path or URL (optional).
        video: Video path or URL (optional).
        task: Optional built-in task hint (caption/temporal/embodied/...).
        port: vLLM server port.
        max_tokens: Output token cap.
        think: Enable explicit reasoning format.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    proc = _reason(prompt, image, video, task, port, max_tokens, think)
    return proc_result(proc, success_text="cosmos3 reason result:",
                       fail_text=f"c3-reason failed: {proc.get('stderr','')[:200]}")


@tool
def cosmos3_caption(video: str = "", image: str = "", port: int = 8000, max_tokens: int = 4096) -> dict:
    """Generate a detailed natural-language caption of a video or image.

    Uses the Cosmos 3 Reasoner to describe the scene, objects, and activity.
    Requires a running reasoner server (``cosmos3_serve`` / ``just c3-serve-reason``).

    Args:
        video: Path or URL to a video to caption (optional if ``image`` is given).
        image: Path or URL to an image to caption (optional if ``video`` is given).
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    proc = _reason("Caption in detail.", image, video, "caption", port, max_tokens, False)
    return proc_result(proc, "cosmos3 caption:", "c3 caption failed")


@tool
def cosmos3_temporal(video: str, port: int = 8000, max_tokens: int = 2048) -> dict:
    """List the notable events in a video with approximate timestamps.

    Temporal localization with the Cosmos 3 Reasoner — useful for summarizing or
    indexing footage. Requires a running reasoner server.

    Args:
        video: Path or URL to the video to analyze.
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    proc = _reason("List the notable events with approximate timestamps.",
                   "", video, "temporal", port, max_tokens, False)
    return proc_result(proc, "cosmos3 temporal:", "c3 temporal failed")


@tool
def cosmos3_embodied(video: str = "", image: str = "", port: int = 8000, max_tokens: int = 1024) -> dict:
    """Predict the next immediate action for an embodied agent from a scene.

    Embodied reasoning with chain-of-thought — given a video or image of a robot
    or agent, suggests the next action to take. Requires a running reasoner server.

    Args:
        video: Path or URL to a video of the scene (optional if ``image`` given).
        image: Path or URL to an image of the scene (optional if ``video`` given).
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    proc = _reason("What can be the next immediate action?",
                   image, video, "embodied", port, max_tokens, True)
    return proc_result(proc, "cosmos3 embodied:", "c3 embodied failed")


@tool
def cosmos3_ground(image: str, object_name: str, port: int = 8000, max_tokens: int = 1024) -> dict:
    """Locate an object in an image and return its 2D bounding box as JSON.

    2D visual grounding with the Cosmos 3 Reasoner. Requires a running reasoner server.

    Args:
        image: Path or URL to the image to search.
        object_name: Natural-language name of the object to locate.
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    proc = _reason("Locate the bounding box of " + object_name + ". Return JSON.",
                   image, "", "grounding", port, max_tokens, False)
    return proc_result(proc, "cosmos3 grounding:", "c3 grounding failed")


@tool
def cosmos3_plausibility(video: str, port: int = 8000, max_tokens: int = 1024) -> dict:
    """Judge whether a video is physically plausible and explain why.

    Checks object permanence, shape constancy, and continuous trajectories, then
    classifies the clip as plausible or implausible with a rationale. Requires a
    running reasoner server.

    Args:
        video: Path or URL to the video to evaluate.
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    proc = _reason(
        "Is this video physically plausible (object permanence, shape "
        "constancy, continuous trajectories)? Answer plausible or "
        "implausible, then explain.",
        "", video, "plausibility", port, max_tokens, True)
    return proc_result(proc, "cosmos3 plausibility:", "c3 plausibility failed")


@tool
def cosmos3_situation(video: str, question: str = "", port: int = 8000, max_tokens: int = 2048) -> dict:
    """Describe the situation in a video and predict the most likely next action.

    Situation awareness with the Cosmos 3 Reasoner. Requires a running reasoner server.

    Args:
        video: Path or URL to the video to analyze.
        question: Optional specific question; if omitted, a general situation
            description and next-action prediction is produced.
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    p = question or "Describe the situation and predict the most likely next action."
    proc = _reason(p, "", video, "situation", port, max_tokens, True)
    return proc_result(proc, "cosmos3 situation:", "c3 situation failed")


@tool
def cosmos3_action_cot(
    image: str = "",
    video: str = "",
    task_instruction: str = "complete the task",
    port: int = 8000,
    max_tokens: int = 2048,
) -> dict:
    """Produce a 2D end-effector trajectory for a task as chain-of-thought JSON.

    Action reasoning with the Cosmos 3 Reasoner — returns a pixel-space trajectory
    (e.g. ``{"point_2d": [x, y], "label": "gripper trajectory"}``) for the given
    task. Requires a running reasoner server.

    Args:
        image: Path or URL to an image of the scene (optional if ``video`` given).
        video: Path or URL to a video of the scene (optional if ``image`` given).
        task_instruction: The task the end effector should accomplish.
        port: Port of the local vLLM reasoner server.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text answer; on error the content holds an
        error message and ``status`` is ``"error"``.
    """
    p = ('You are given the task "' + task_instruction + '". Specify the 2D '
         "trajectory your end effector should follow in pixel space. Return JSON "
         'like {"point_2d": [x, y], "label": "gripper trajectory"}.')
    proc = _reason(p, image, video, "action_cot", port, max_tokens, True)
    return proc_result(proc, "cosmos3 action_cot:", "c3 action_cot failed")


# ----- Generator ----------------------------------------------------------
@tool
def cosmos3_text2image(prompt: str, out: str = "/tmp/c3_image.png", steps: int = 35,
                       guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Generate an image from a text prompt (Cosmos 3 Generator, in-process Diffusers).

    Runs locally with Diffusers — no server required.

    Args:
        prompt: Text description of the image to generate.
        out: Output PNG file path to write.
        steps: Diffusion denoising steps (higher = slower, more detail).
        guidance: Classifier-free guidance scale.
        res: Output resolution preset ("480", "720", or "1080").
        seed: Random seed for reproducible generation.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content reports the output file path; on error ``status`` is ``"error"``.
    """
    proc = _gen("text2image", prompt, "", out, 1, 24, steps, guidance, res, "false", seed)
    return proc_result(proc, "cosmos3 text2image -> " + out, "c3 text2image failed")


@tool
def cosmos3_text2video(prompt: str, out: str = "/tmp/c3_t2v.mp4", frames: int = 189,
                       fps: int = 24, steps: int = 35, guidance: float = 6.0,
                       res: str = "480", seed: int = 0) -> dict:
    """Generate a video from a text prompt (Cosmos 3 Generator, in-process Diffusers).

    Runs locally with Diffusers — no server required.

    Args:
        prompt: Text description of the content to generate.
        out: Output file path to write.
        frames: Number of frames to generate.
        fps: Frames per second of the output video.
        steps: Diffusion denoising steps (higher = slower, more detail).
        guidance: Classifier-free guidance scale.
        res: Output resolution preset ("480", "720", or "1080").
        seed: Random seed for reproducible generation.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content reports the output file path; on error ``status`` is ``"error"``.
    """
    proc = _gen("text2video", prompt, "", out, frames, fps, steps, guidance, res, "false", seed)
    return proc_result(proc, "cosmos3 text2video -> " + out, "c3 text2video failed")


@tool
def cosmos3_image2video(prompt: str, image: str, out: str = "/tmp/c3_i2v.mp4",
                        frames: int = 189, fps: int = 24, steps: int = 35,
                        guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Animate a still image into a video guided by a text prompt (Cosmos 3 Generator).

    Image-conditioned video generation, locally via Diffusers — no server required.

    Args:
        prompt: Text description of the motion / how the scene should evolve.
        image: Path to the conditioning image (the first frame).
        out: Output MP4 file path to write.
        frames: Number of frames to generate.
        fps: Frames per second of the output video.
        steps: Diffusion denoising steps (higher = slower, more detail).
        guidance: Classifier-free guidance scale.
        res: Output resolution preset ("480", "720", or "1080").
        seed: Random seed for reproducible generation.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content reports the output file path; on error ``status`` is ``"error"``.
    """
    proc = _gen("image2video", prompt, image, out, frames, fps, steps, guidance, res, "false", seed)
    return proc_result(proc, "cosmos3 image2video -> " + out, "c3 image2video failed")


@tool
def cosmos3_text2video_sound(prompt: str, out: str = "/tmp/c3_t2v_sound.mp4",
                             frames: int = 189, fps: int = 24, steps: int = 35,
                             guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Generate a video with a synchronized soundtrack from a text prompt.

    Cosmos 3 Generator with audio (stereo AAC @ 48kHz), locally via Diffusers — no
    server required. Needs a sound-capable checkpoint (Cosmos3-Nano).

    Args:
        prompt: Text description of the content to generate.
        out: Output file path to write.
        frames: Number of frames to generate.
        fps: Frames per second of the output video.
        steps: Diffusion denoising steps (higher = slower, more detail).
        guidance: Classifier-free guidance scale.
        res: Output resolution preset ("480", "720", or "1080").
        seed: Random seed for reproducible generation.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content reports the output file path; on error ``status`` is ``"error"``.
    """
    proc = _gen("text2video-with-sound", prompt, "", out, frames, fps, steps, guidance, res, "true", seed)
    return proc_result(proc, "cosmos3 text2video+sound -> " + out, "c3 t2v-sound failed")


@tool
def cosmos3_image2video_sound(prompt: str, image: str, out: str = "/tmp/c3_i2v_sound.mp4",
                              frames: int = 189, fps: int = 24, steps: int = 35,
                              guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Animate an image into a video with a synchronized soundtrack.

    Image-conditioned motion with synchronized stereo AAC @ 48kHz audio, locally
    via Diffusers (no server). Needs a sound-capable checkpoint (Cosmos3-Nano).

    Args:
        prompt: Text description of the motion / how the scene should evolve.
        image: Path to the conditioning image (the first frame).
        out: Output MP4 file path to write.
        frames: Number of frames to generate.
        fps: Frames per second of the output video.
        steps: Diffusion denoising steps (higher = slower, more detail).
        guidance: Classifier-free guidance scale.
        res: Output resolution preset ("480", "720", or "1080").
        seed: Random seed for reproducible generation.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the tool's output; on error ``status`` is ``"error"`` with a message.
    """
    proc = _gen("image2video-with-sound", prompt, image, out, frames, fps, steps, guidance, res, "true", seed)
    return proc_result(proc, "cosmos3 image2video+sound -> " + out, "c3 i2v-sound failed")


@tool
def cosmos3_video2video(
    video: str,
    prompt: str,
    out: str = "/tmp/c3_v2v.mp4",
    port: int = 8001,
    steps: int = 35,
    guidance: float = 8.0,
    size: str = "832x480",
    frames: int = 29,
    fps: int = 16,
    seed: int = 0,
    negative: str = "blurry, distorted, low quality",
    guardrails: bool = True,
    condition_frames: str = "0",
    condition_keep: str = "last",
    generate_sound: bool = False,
    max_sequence_length: int = 512,
) -> dict:
    """Cosmos 3 video-to-video: re-render an input video with a new prompt.

    Structure-preserving transfer (day->night, recolor, restyle, change the scene)
    via the vLLM-Omni server's /v1/videos/sync endpoint. Start the server first
    with `just c3-omni-docker` (Docker image vllm/vllm-omni:cosmos3 — the only
    build with all modalities incl. video2video).

    How much the prompt changes the video is controlled by the conditioning:
    fewer/earlier conditioning frames + higher guidance = a stronger transform.

    Args:
        video: Path to the input video (local file).
        prompt: Target description (the transformation to apply). Be emphatic and
            pair with a `negative` prompt for strong restyles (e.g. day->night).
        out: Output MP4 path.
        port: Omni server port (default 8001).
        steps: Diffusion steps (35 recommended for a clean restyle).
        guidance: CFG scale. 6 = subtle/structure-faithful; 8-12 = strong restyle.
        size: Output resolution "WxH".
        frames: Frame count.
        fps: Frames per second.
        seed: Reproducibility seed.
        negative: Negative prompt (helps push away the original look).
        guardrails: Enable Cosmos 3 safety guardrails.
        condition_frames: Latent frame indexes kept as clean conditioning, as a
            comma-separated string. Default "0" (anchor only the first latent ->
            strongest transform). The model default is "0,1" (more faithful, weaker
            change). More indexes => closer to the original video.
        condition_keep: Which end of the clip the conditioning frames come from:
            "first" or "last" (default "last").
        generate_sound: Produce a synchronized soundtrack (stereo AAC@48kHz)
            alongside the transformed video (video-to-video-with-sound).
        max_sequence_length: Max prompt tokens kept for conditioning (Cosmos 3
            default 512); longer prompts are truncated with a warning.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the transformed video's output path and byte size; on error ``status`` is ``"error"`` with a message.
    """
    import json as _json
    import os as _os

    try:
        import requests
    except ImportError:
        return err("requests required for cosmos3_video2video: pip install requests")

    # Confine the LLM-supplied input video to the workspace (CWE-22).
    try:
        path = str(resolve_in_workspace(video, must_exist=True))
    except SecurityError as e:
        return err(str(e))

    if condition_keep not in ("first", "last"):
        return err("condition_keep must be 'first' or 'last'")
    try:
        cond_idx = [int(x.strip()) for x in str(condition_frames).split(",") if x.strip() != ""]
        if not cond_idx:
            raise ValueError
    except ValueError:
        return err("condition_frames must be comma-separated non-negative ints, e.g. '0' or '0,1'")

    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": guardrails,
        "condition_frame_indexes_vision": cond_idx,
        "condition_video_keep": condition_keep,
    }
    if generate_sound:
        extra["generate_sound"] = True
    data = {
        "prompt": prompt,
        "negative_prompt": negative,
        "size": size,
        "num_frames": str(frames),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance),
        "flow_shift": "10.0",
        "seed": str(seed),
        "max_sequence_length": str(max_sequence_length),
        "extra_params": _json.dumps(extra),
    }
    if generate_sound:
        data["generate_sound"] = "true"
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"http://localhost:{port}/v1/videos/sync",
                data=data,
                files={"input_reference": (_os.path.basename(path), f, "video/mp4")},
                timeout=60 * 30,
            )
        if resp.status_code != 200:
            return err(f"omni server returned {resp.status_code}: {resp.text[:200]}")
        # Confine the LLM-supplied output path to the workspace (CWE-22).
        try:
            out = str(resolve_output_path(out))
        except SecurityError as e:
            return err(str(e))
        with open(out, "wb") as g:
            g.write(resp.content)
        return ok(
            f"cosmos3 video2video -> {out} ({len(resp.content)} bytes) "
            f"[guidance={guidance}, condition_frames={cond_idx}, keep={condition_keep}, "
            f"sound={generate_sound}]",
            data={"out": out, "bytes": len(resp.content),
                  "condition_frame_indexes_vision": cond_idx, "condition_video_keep": condition_keep,
                  "generate_sound": generate_sound},
        )
    except Exception as e:
        return err("video2video request failed: " + str(e))


# ----- Action / World-Model -----------------------------------------------
# All action tools take a JSONL spec (one line per run). The spec's `model_mode`
# field selects forward_dynamics / inverse_dynamics / policy. These thin wrappers
# call `just c3-action <input_jsonl>` (Cosmos Framework via torchrun). Sample
# specs + assets live in the cosmos repo cookbooks/cosmos3/generator/action.
@tool
def cosmos3_forward_dynamics(input_jsonl: str, out: str = "/tmp/c3_fd",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Forward dynamics: start image + action chunk -> future video (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="forward_dynamics", vision_path,
            action_path, domain_name, action_chunk_size, fps, image_size, name.
        out: output dir (writes <out>/<name>/vision.mp4).
        checkpoint: Cosmos 3 checkpoint name.
        seed: reproducibility seed.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the generated future-video output directory; on error ``status`` is ``"error"`` with a message.
    """
    try:
        proc = _action(input_jsonl, out, checkpoint, seed)
    except SecurityError as e:
        return err(str(e))
    return proc_result(proc, "cosmos3 forward_dynamics -> " + out, "c3 fd failed")


@tool
def cosmos3_inverse_dynamics(input_jsonl: str, out: str = "/tmp/c3_id",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Inverse dynamics: video + instruction -> predicted action chunk (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="inverse_dynamics", vision_path
            (input video), domain_name, name.
        out: output dir.
        checkpoint: Cosmos 3 checkpoint name.
        seed: reproducibility seed.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the predicted action-chunk output directory; on error ``status`` is ``"error"`` with a message.
    """
    try:
        proc = _action(input_jsonl, out, checkpoint, seed)
    except SecurityError as e:
        return err(str(e))
    return proc_result(proc, "cosmos3 inverse_dynamics -> " + out, "c3 id failed")


@tool
def cosmos3_policy(input_jsonl: str, out: str = "/tmp/c3_policy",
                  checkpoint: str = "Cosmos3-Nano-Policy-DROID", seed: int = 0) -> dict:
    """Action policy: image + instruction -> action chunk + rollout video (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="policy", vision_path, domain_name
            (e.g. bridge_orig_lerobot), prompt (instruction), name.
        out: output dir.
        checkpoint: Cosmos 3 policy checkpoint (default Cosmos3-Nano-Policy-DROID).
        seed: reproducibility seed.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the path to the action-chunk and rollout-video output directory; on error ``status`` is ``"error"`` with a message.
    """
    try:
        proc = _action(input_jsonl, out, checkpoint, seed)
    except SecurityError as e:
        return err(str(e))
    return proc_result(proc, "cosmos3 policy -> " + out, "c3 policy failed")


# ----- Server lifecycle ---------------------------------------------------
@tool
def cosmos3_serve(action: str = "status", surface: str = "reason",
                  model: str = "nvidia/Cosmos3-Nano", port: int = 0, tp: int = 1) -> dict:
    """Manage local Cosmos 3 servers (no NIM).

    Args:
        action: start | stop | status
        surface: reason (vLLM) | omni (vLLM-Omni generator)
        model: HF model id to serve.
        port: server port (0 = recipe default).
        tp: tensor-parallel size (reason only).

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the server's start/stop/status output; on error ``status`` is ``"error"`` with a message.
    """
    if action == "status":
        return proc_result(just_run("c3-serve-status"), "c3 server status:", "status failed")
    if surface == "reason":
        if action == "start":
            args = ["c3-serve-reason", model]
            if port:
                args += [str(port), str(tp)]
            return proc_result(just_run(*args, timeout_s=120), "c3 reason server starting", "start failed")
        if action == "stop":
            return proc_result(just_run("c3-serve-stop-reason"), "c3 reason server stopped", "stop failed")
    elif surface == "omni":
        if action == "start":
            args = ["c3-serve-omni", model]
            if port:
                args.append(str(port))
            return proc_result(just_run(*args, timeout_s=120), "c3 omni server starting", "start failed")
        if action == "stop":
            return proc_result(just_run("c3-serve-stop-omni"), "c3 omni server stopped", "stop failed")
    return err("unknown action/surface: " + action + "/" + surface)
