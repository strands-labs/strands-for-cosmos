# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos 3 Reasoner model provider for Strands Agents.

Cosmos 3 is an omnimodal world model. The **Reasoner** surface produces text
from text + vision (image/video) inputs. Unlike Cosmos-Reason2 (direct Qwen3VL
via Transformers), the Cosmos 3 Transformers reasoner path is "coming soon"
upstream, so this provider talks to a **local vLLM OpenAI-compatible server**
running the `Cosmos3ReasonerForConditionalGeneration` architecture.

Start a server (see justfile `c3-serve-reason`):

    CUDA_VISIBLE_DEVICES=0 vllm serve nvidia/Cosmos3-Nano \\
      --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \\
      --mm-encoder-tp-mode data --async-scheduling \\
      --allowed-local-media-path / \\
      --media-io-kwargs '{"video": {"num_frames": -1}}' --port 8000

Then:

    >>> from strands import Agent
    >>> from strands_cosmos import Cosmos3ReasonerModel
    >>> model = Cosmos3ReasonerModel(base_url="http://localhost:8000/v1")
    >>> agent = Agent(model=model)
    >>> agent("Caption in detail: <video>scene.mp4</video>")

- Cosmos 3: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- Models: https://huggingface.co/nvidia/Cosmos3-Nano
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    cast,
)

from strands.models._validation import (
    validate_config_keys,
    warn_on_tool_choice_not_supported,
)
from strands.models.model import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import TypedDict, Unpack, override

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Reasoner sampling presets from the Cosmos 3 README.
SAMPLING_NO_THINK = {
    "top_p": 0.8,
    "temperature": 0.7,
    "presence_penalty": 1.5,
    "extra_body": {"top_k": 20, "repetition_penalty": 1.0},
}
SAMPLING_THINK = {
    "top_p": 0.95,
    "temperature": 0.6,
    "presence_penalty": 0.0,
    "extra_body": {"top_k": 20, "repetition_penalty": 1.0},
}

REASONING_SUFFIX = (
    "\n\nAnswer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)

# Built-in reasoner task prompts (Cosmos 3 Reasoner capabilities).
TASK_PROMPTS = {
    "caption": "Caption the video in detail.",
    "temporal": "List the notable events with approximate timestamps.",
    "embodied": "What can be the next immediate action?",
    "plausibility": (
        "Is this video physically plausible according to your understanding of "
        "object permanence, shape constancy, and continuous trajectories? "
        "Answer with a single label: plausible or implausible, then explain."
    ),
    "situation": "Describe the situation and predict the most likely next action.",
    "grounding": "Locate the bounding box of {object_name}. Return JSON.",
    "describe": (
        "Caption the notable attributes for the marked subjects. Return JSON with "
        'keys "subject_id", "category", "caption".'
    ),
    "action_cot": (
        'You are given the task "{task_instruction}". Specify the 2D trajectory your '
        "end effector should follow in pixel space. Return JSON like "
        '{{"point_2d": [x, y], "label": "gripper trajectory"}}.'
    ),
    "driving": (
        "The video is the observation from the vehicle's camera. Think step by step "
        "and identify objects critical for safe navigation."
    ),
}


def _media_to_url(path_or_url: str) -> str:
    """Resolve a media reference to a URL the vLLM server accepts.

    ``http(s)://`` URLs are validated and passed through; local files are
    confined to the project workspace and encoded as a base64 data URI (no
    ``file://`` fallback, so the server cannot read arbitrary local paths).
    """
    from .tools._security import (
        resolve_in_workspace,
        validate_url,
    )

    if path_or_url.startswith(("http://", "https://")):
        # Remote media the server fetches: allow public hosts but block
        # private/link-local/metadata SSRF targets (CWE-918).
        return validate_url(path_or_url, allow_public=True)
    if path_or_url.startswith("data:"):
        return path_or_url
    # Local file: confine to the workspace before reading (no file:// escape).
    p = resolve_in_workspace(path_or_url, must_exist=True)
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"
    with open(p, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class Cosmos3ReasonerModel(Model):
    """NVIDIA Cosmos 3 Reasoner provider (vLLM OpenAI backend).

    Multimodal text-output reasoning over images and video: captioning, temporal
    localization, embodied next-action, 2D grounding, physical plausibility,
    situation understanding, describe-anything, and action chain-of-thought.

    Inline media tags supported in user messages:
        - <video>path/or/url.mp4</video>
        - <image>path/or/url.jpg</image>
    """

    class Cosmos3ReasonerConfig(TypedDict, total=False):
        """Cosmos 3 Reasoner configuration."""

        model_id: str
        base_url: str
        api_key: str
        params: Optional[Dict[str, Any]]
        reasoning: bool
        max_tokens: int
        seed: Optional[int]
        # video frame sampling, e.g. {"video": {"fps": 4.0}} or {"video": {"num_frames": 16}}
        media_io_kwargs: Optional[Dict[str, Any]]
        # per-image resize bounds, e.g. {"size": {"shortest_edge": 1568}}
        mm_processor_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        **model_config: Unpack[Cosmos3ReasonerConfig],
    ) -> None:
        """Initialize the Cosmos 3 Reasoner provider.

        Args:
            model_id: HF model id served by vLLM (default nvidia/Cosmos3-Nano).
                      If the server reports a different id, it's auto-resolved.
            base_url: vLLM OpenAI-compatible base URL.
            **model_config: Additional config (reasoning, max_tokens, seed, etc.).
        """
        validate_config_keys(model_config, self.Cosmos3ReasonerConfig)

        self.config: Dict[str, Any] = {
            "model_id": model_id,
            "base_url": base_url,
            "api_key": os.getenv("COSMOS3_API_KEY", "EMPTY"),
            "reasoning": False,
            "max_tokens": 4096,
            "seed": 0,
            "media_io_kwargs": None,
            "mm_processor_kwargs": None,
        }
        self.config.update(model_config)
        logger.debug("config=<%s> | initializing cosmos3 reasoner", self.config)

        self._client = None
        self._resolved_model_id: Optional[str] = None

    # ── Strands Model API ─────────────────────────────────────────────────
    @override
    def update_config(self, **model_config: Unpack[Cosmos3ReasonerConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.Cosmos3ReasonerConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> Cosmos3ReasonerConfig:  # type: ignore[override]
        return cast(Cosmos3ReasonerModel.Cosmos3ReasonerConfig, self.config)

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package required for Cosmos3ReasonerModel. "
                    "pip install openai"
                ) from e
            self._client = openai.OpenAI(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
            )
        return self._client

    def _resolve_model_id(self) -> str:
        """Resolve the served model id dynamically (vLLM serves one model)."""
        if self._resolved_model_id:
            return self._resolved_model_id
        try:
            client = self._get_client()
            models = client.models.list()
            if models.data:
                self._resolved_model_id = models.data[0].id
                return self._resolved_model_id
        except Exception as e:
            logger.debug("could not list models, using configured id: %s", e)
        self._resolved_model_id = self.config["model_id"]
        return self._resolved_model_id

    def _extract_media_to_openai(self, messages: Messages) -> List[Dict[str, Any]]:
        """Convert Strands messages → OpenAI chat content with media URLs.

        Parses native image/video ContentBlocks (preferred) and, for backward
        compatibility, inline <image>/<video> tags. All media routes through
        the same hardened `_media_to_url` resolver.
        """
        oai_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            parts: List[Dict[str, Any]] = []

            for content in message["content"]:
                if "text" in content:
                    text = content["text"]

                    # <image>...</image>
                    for m in re.findall(r"<image>(.*?)</image>", text):
                        parts.append(
                            {"type": "image_url", "image_url": {"url": _media_to_url(m.strip())}}
                        )
                    text = re.sub(r"<image>.*?</image>", "", text)

                    # <video>...</video>
                    for m in re.findall(r"<video>(.*?)</video>", text):
                        parts.append(
                            {"type": "video_url", "video_url": {"url": _media_to_url(m.strip())}}
                        )
                    text = re.sub(r"<video>.*?</video>", "", text)

                    cleaned = text.strip()
                    if cleaned:
                        parts.append({"type": "text", "text": cleaned})

                elif "image" in content:
                    img = content["image"]
                    source = img.get("source", {})
                    if "bytes" in source:
                        fmt = img.get("format", "jpeg")
                        b64 = base64.b64encode(source["bytes"]).decode("utf-8")
                        url = f"data:image/{fmt};base64,{b64}"
                    else:
                        url = _media_to_url(source.get("url", ""))
                    parts.append({"type": "image_url", "image_url": {"url": url}})

                elif "video" in content:
                    # Native Strands video ContentBlock: {"source": {...}, "format": "mp4"}
                    # Mirrors the image path so video flows through the SAME hardened
                    # media resolver (workspace-confine + base64 / SSRF allow-list)
                    # instead of the old <video> tag-string-in-prompt shape.
                    vid = content["video"]
                    source = vid.get("source", {})
                    if "bytes" in source:
                        fmt = vid.get("format", "mp4")
                        b64 = base64.b64encode(source["bytes"]).decode("utf-8")
                        url = f"data:video/{fmt};base64,{b64}"
                    else:
                        url = _media_to_url(source.get("url", ""))
                    parts.append({"type": "video_url", "video_url": {"url": url}})

            if parts:
                # OpenAI requires non-empty content; collapse single text to str
                if len(parts) == 1 and parts[0].get("type") == "text":
                    oai_messages.append({"role": role, "content": parts[0]["text"]})
                else:
                    oai_messages.append({"role": role, "content": parts})

        return oai_messages

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Cosmos 3 Reasoner generation (text out) via vLLM."""
        if tool_choice:
            warn_on_tool_choice_not_supported(tool_choice)

        oai_messages = self._extract_media_to_openai(messages)

        if system_prompt:
            oai_messages.insert(0, {"role": "system", "content": system_prompt})

        # Explicit reasoning: append <think> format instruction to last user msg.
        if self.config.get("reasoning") and oai_messages:
            for msg in reversed(oai_messages):
                if msg["role"] == "user":
                    if isinstance(msg["content"], str):
                        msg["content"] += REASONING_SUFFIX
                    else:
                        msg["content"].append({"type": "text", "text": REASONING_SUFFIX})
                    break

        preset = SAMPLING_THINK if self.config.get("reasoning") else SAMPLING_NO_THINK
        params = self.config.get("params") or {}

        extra_body = dict(preset["extra_body"])
        if self.config.get("media_io_kwargs"):
            extra_body["media_io_kwargs"] = self.config["media_io_kwargs"]
        if self.config.get("mm_processor_kwargs"):
            extra_body["mm_processor_kwargs"] = self.config["mm_processor_kwargs"]

        request: Dict[str, Any] = {
            "model": self._resolve_model_id(),
            "messages": oai_messages,
            "max_tokens": params.get("max_tokens", self.config["max_tokens"]),
            "top_p": params.get("top_p", preset["top_p"]),
            "temperature": params.get("temperature", preset["temperature"]),
            "presence_penalty": params.get("presence_penalty", preset["presence_penalty"]),
            "stream": True,
            "extra_body": extra_body,
        }
        if self.config.get("seed") is not None:
            request["seed"] = self.config["seed"]

        client = self._get_client()

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        token_count = 0
        try:
            response = client.chat.completions.create(**request)
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield {"contentBlockDelta": {"delta": {"text": delta.content}}}
                    token_count += 1
        except Exception as e:
            logger.error("cosmos3 reasoner stream error: %s", e)
            yield {"contentBlockDelta": {"delta": {"text": f"[Cosmos3 error: {e}]"}}}

        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": token_count, "totalTokens": token_count},
                "metrics": {"latencyMs": 0},
            }
        }

    @override
    async def structured_output(
        self,
        output_model: Any,
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Structured output via JSON-instruction prompting."""
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON matching:\n{json.dumps(schema, indent=2)}"
        augmented = (system_prompt or "") + json_instruction

        response_text = ""
        async for event in self.stream(prompt, system_prompt=augmented, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            yield event

        try:
            txt = response_text
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0]
            data = json.loads(txt.strip())
            yield {"output": output_model(**data)}
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {e}\n{response_text}") from e
