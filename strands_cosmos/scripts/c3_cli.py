# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos 3 CLI entrypoints (reason / gen / upsample).

Backs the one-line justfile recipes ``c3-reason``, ``c3-gen``, ``c3-upsample``.

Each subcommand drives **our own model providers**
(``Cosmos3ReasonerModel`` / ``Cosmos3GeneratorModel``) -- it does NOT open a
raw OpenAI client. This keeps a single implementation/entrypoint: the operator
CLI, the agent SDK path, and these recipes all share the same provider, media
resolver, sampling presets, and security model.

All untrusted input arrives via environment variables (set by the recipe from
positional/validated params) and is read with ``os.environ`` here -- never
interpolated into a shell template. This preserves the CWE-78 hardening that
the original inline heredocs carried. Media refs route through the provider's
hardened ``_media_to_url`` resolver (workspace allow-list + SSRF, CWE-22/918).

Usage (invoked by justfile):
    python -m strands_cosmos.scripts.c3_cli reason
    python -m strands_cosmos.scripts.c3_cli gen
    python -m strands_cosmos.scripts.c3_cli upsample
    python -m strands_cosmos.scripts.c3_cli infer
    python -m strands_cosmos.scripts.c3_cli v2v
"""
from __future__ import annotations

import asyncio
import os
import sys


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, "true" if default else "false").lower() == "true"


def _run_reasoner(model, messages, system_prompt=None) -> str:
    """Drive our Cosmos3ReasonerModel.stream() and collect the text output.

    Uses the provider directly (no Agent, no OpenAI client) so the recipe and
    the SDK share one code path.
    """

    async def _collect() -> str:
        chunks: list[str] = []
        async for event in model.stream(messages, system_prompt=system_prompt):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                chunks.append(delta["text"])
        return "".join(chunks)

    return asyncio.run(_collect())


# ── reason: one-shot reasoning via our Cosmos3ReasonerModel ────────────────
def reason() -> int:
    from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel

    prompt = os.environ.get("C3_PROMPT", "")
    image = os.environ.get("C3_IMAGE", "")
    video = os.environ.get("C3_VIDEO", "")
    think = _env_bool("C3_THINK")
    port = os.environ.get("C3_PORT", "8000")
    max_tokens = _env_int("C3_MAX_TOKENS", 4096)

    model = Cosmos3ReasonerModel(
        base_url=f"http://localhost:{port}/v1",
        reasoning=think,  # provider appends the <think>/<answer> suffix + THINK preset
        max_tokens=max_tokens,
        seed=0,
    )

    # Build native Strands content blocks; media routes through the provider's
    # hardened resolver via {"image"|"video": {"source": {"url": ...}}}.
    content: list[dict] = []
    if image:
        content.append({"image": {"source": {"url": image}}})
    if video:
        content.append({"video": {"source": {"url": video}}})
    content.append({"text": prompt})

    out = _run_reasoner(model, [{"role": "user", "content": content}])
    print(out)
    return 0


# ── gen: in-proc Diffusers generation via our Cosmos3GeneratorModel ────────
def gen() -> int:
    from strands_cosmos.cosmos3_generator_model import Cosmos3GeneratorModel

    allowed_modes = {
        "text2image",
        "text2video",
        "image2video",
        "text2video-with-sound",
        "image2video-with-sound",
    }
    mode = os.environ.get("C3_GEN_MODE", "text2video")
    if mode not in allowed_modes:
        print(f"invalid mode: {mode!r}", file=sys.stderr)
        return 2

    prompt = os.environ.get("C3_GEN_PROMPT", "")
    image = os.environ.get("C3_GEN_IMAGE", "") or None
    out = os.environ.get("C3_GEN_OUT", "/tmp/c3_out.mp4")
    model_id = os.environ.get("C3_GEN_MODEL", "nvidia/Cosmos3-Nano")

    m = Cosmos3GeneratorModel(model_id=model_id)
    out = m.generate(
        mode=mode,
        prompt=prompt,
        out_path=out,
        image=image,
        num_frames=_env_int("C3_GEN_FRAMES", 189),
        fps=_env_int("C3_GEN_FPS", 24),
        num_inference_steps=_env_int("C3_GEN_STEPS", 35),
        guidance_scale=_env_float("C3_GEN_GUIDANCE", 6.0),
        resolution=os.environ.get("C3_GEN_RES", "480"),
        enable_sound=_env_bool("C3_GEN_SOUND"),
        seed=_env_int("C3_GEN_SEED", 0),
    )
    print(out)
    return 0


# ── upsample: expand a short description into a dense generator prompt ──────
# Prompt construction is Cosmos-Framework logic (build_messages); the LLM call
# itself is routed through OUR Cosmos3ReasonerModel rather than a raw client.
def upsample() -> int:
    from cosmos_framework.model.vfm.upsampler.prompts import (
        build_messages,
        clean_response,
    )

    from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel

    task = os.environ.get("C3_UP_TASK", "t2v")
    desc = os.environ.get("C3_UP_DESC", "")
    port = os.environ.get("C3_UP_PORT", "8000")
    image = os.environ.get("C3_UP_IMAGE", "")

    kw = dict(
        task=task,
        description=desc,
        aspect_ratio=os.environ.get("C3_UP_ASPECT", "16,9"),
        resolution_w=_env_int("C3_UP_WIDTH", 832),
        resolution_h=_env_int("C3_UP_HEIGHT", 480),
    )
    if task in ("t2v", "i2v"):
        kw.update(
            fps=_env_int("C3_UP_FPS", 24),
            duration_secs=_env_int("C3_UP_DURATION", 8),
        )

    fw_messages = build_messages(**kw)

    # Framework returns OpenAI-style [system, user]. Translate to Strands:
    #   - system message  -> system_prompt
    #   - user message    -> a Strands user message (text + optional image)
    system_prompt = None
    user_text = ""
    for msg in fw_messages:
        role = msg.get("role")
        c = msg.get("content")
        text = c if isinstance(c, str) else ""
        if role == "system":
            system_prompt = text
        elif role == "user":
            user_text = text

    content: list[dict] = []
    # i2v: attach the conditioning image (provider resolves it via _media_to_url).
    if task == "i2v" and image:
        content.append({"image": {"source": {"url": image}}})
    content.append({"text": user_text})

    # Upsampler sampling == our NO_THINK preset (top_p=0.8, temp=0.7,
    # presence_penalty=1.5, top_k=20); min_p=0.0 / repetition_penalty=1.0 are
    # both no-ops, so the preset is functionally identical. Only seed (3407)
    # and the larger token budget differ -> pass them via config.
    model = Cosmos3ReasonerModel(
        base_url=f"http://localhost:{port}/v1",
        reasoning=False,
        max_tokens=20000,
        seed=3407,
    )

    text = _run_reasoner(model, [{"role": "user", "content": content}], system_prompt).strip()
    cleaned, _ = clean_response(text)
    cleaned = cleaned.removeprefix("```json\n").removesuffix("```").strip()
    print(cleaned)
    return 0



# ── infer: TRT-Edge VLM inference via our cosmos_inference tool ────────────
# Routes the `infer` recipe through the SAME hardened tool the agent uses
# (SSRF-guarded endpoint + workspace-confined image path), instead of a raw
# curl/jq OpenAI-protocol call duplicated in the justfile.
def infer() -> int:
    from strands_cosmos.tools.inference import cosmos_inference

    image = os.environ.get("C3_INFER_IMAGE", "")
    prompt = os.environ.get("C3_INFER_PROMPT", "describe the scene")
    url = os.environ.get("C3_INFER_URL", "")
    max_tokens = _env_int("C3_INFER_MAX_TOKENS", 256)
    temperature = _env_float("C3_INFER_TEMP", 0.2)

    # Call the tool's underlying function directly (bypass the @tool wrapper).
    fn = getattr(cosmos_inference, "_tool_func", cosmos_inference)
    result = fn(
        prompt=prompt,
        image_path=image,
        server_url=url,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Normalize the Strands tool-result dict to stdout text + exit code.
    status = result.get("status")
    for block in result.get("content", []):
        if "text" in block:
            print(block["text"])
    return 0 if status == "success" else 1



# ── v2v: vLLM-Omni video-to-video via our cosmos3_video2video tool ─────────
# Routes the `c3-v2v` recipe through the SAME hardened tool the agent uses
# (workspace-confined input + output, condition-frame validation), instead of
# a raw curl call to /v1/videos/sync duplicated in the justfile.
def v2v() -> int:
    from strands_cosmos.tools.cosmos3 import cosmos3_video2video

    fn = getattr(cosmos3_video2video, "_tool_func", cosmos3_video2video)
    result = fn(
        video=os.environ.get("C3_V2V_INPUT", ""),
        prompt=os.environ.get("C3_V2V_PROMPT", ""),
        out=os.environ.get("C3_V2V_OUT", "/tmp/omni-work/v2v_out.mp4"),
        port=_env_int("C3_V2V_PORT", 8001),
        steps=_env_int("C3_V2V_STEPS", 35),
        guidance=_env_float("C3_V2V_GUIDANCE", 8.0),
        size=os.environ.get("C3_V2V_SIZE", "832x480"),
        frames=_env_int("C3_V2V_FRAMES", 29),
        fps=_env_int("C3_V2V_FPS", 16),
        seed=_env_int("C3_V2V_SEED", 0),
        negative=os.environ.get("C3_V2V_NEGATIVE", "blurry, distorted, low quality"),
        condition_frames=os.environ.get("C3_V2V_COND_FRAMES", "0"),
        condition_keep=os.environ.get("C3_V2V_COND_KEEP", "last"),
    )
    status = result.get("status")
    for block in result.get("content", []):
        if "text" in block:
            print(block["text"])
    return 0 if status == "success" else 1


_DISPATCH = {"reason": reason, "gen": gen, "upsample": upsample, "infer": infer, "v2v": v2v}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _DISPATCH:
        print(
            f"usage: python -m strands_cosmos.scripts.c3_cli {{{'|'.join(_DISPATCH)}}}",
            file=sys.stderr,
        )
        return 2
    # sys.path shim so `import strands_cosmos` works when run from a repo cwd
    sys.path.insert(0, ".")
    return _DISPATCH[argv[0]]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
