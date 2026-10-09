# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos3-Edge reasoner, **in-process** via Transformers (no vLLM server).

``nvidia/Cosmos3-Edge`` is the 4B member of the Cosmos 3 family. Unlike Nano,
its understanding tower loads natively with ``transformers>=5.19``
(``Cosmos3EdgeForConditionalGeneration``), so an edge box such as a Jetson AGX
Thor can run the reasoner without a vLLM server:

    >>> from strands import Agent
    >>> from strands_cosmos import Cosmos3EdgeHFModel
    >>> agent = Agent(model=Cosmos3EdgeHFModel())
    >>> agent("Caption in detail: <video>sample.mp4</video>")

Same ``<image>…</image>`` / ``<video>…</video>`` tag convention as
:class:`~strands_cosmos.CosmosVisionModel`, whose message handling and
streaming this class reuses. Edge answers open with a ``<think>…</think>``
block; by default it is streamed as Strands ``reasoningContent`` and only the
answer is streamed as text (``show_thinking=True`` keeps the raw stream).

MEASURED on a Jetson AGX Thor (arm64, sm_110, torch 2.14.1+cu130, transformers
5.19.0, Edge rev 344d602b): load 9.4-9.8 s, 4.55 GiB bf16 weights; image
caption 15-20 tok/s, peak 4.60 GiB; video caption (sample.mp4, 2,943 input
tokens) 16.7 tok/s, peak 5.18 GiB. Video decoding needs ``torchcodec`` (or
``av``) installed. See docs/guide/cosmos3-edge.md.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional

from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import Unpack, override

from .cosmos3_models import EDGE, get_model, resolve_model_id
from .cosmos_vision_model import CosmosVisionModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = EDGE
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _version_at_least(have: str, want: str) -> bool:
    def parts(v: str):
        out = []
        for p in v.split("+")[0].split("."):
            digits = "".join(ch for ch in p if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return out
    return parts(have) >= parts(want)


class Cosmos3EdgeHFModel(CosmosVisionModel):
    """Cosmos3-Edge understanding tower through ``transformers`` (in-process)."""

    class Cosmos3EdgeHFConfig(CosmosVisionModel.CosmosVisionConfig, total=False):
        show_thinking: bool

    # the parent validates against ``self.CosmosVisionConfig``
    CosmosVisionConfig = Cosmos3EdgeHFConfig  # type: ignore[assignment]

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        **model_config: Unpack[Cosmos3EdgeHFConfig],
    ) -> None:
        """Initialise without loading weights (loaded on first ``stream``).

        Args:
            model_id: ``nvidia/Cosmos3-Edge`` (default; aliases ``edge`` /
                ``cosmos3-edge`` accepted) or a local path / fine-tune of it.
            **model_config: ``params`` (``max_tokens``, ``temperature``, ``top_p``),
                ``device_map`` (default ``"auto"``), ``torch_dtype`` (default
                bfloat16), ``reasoning`` (append the think/answer format prompt),
                ``show_thinking`` (default False: route ``<think>`` to
                ``reasoningContent``), ``fps`` for video sampling.
        """
        from strands.models._validation import validate_config_keys

        from .cosmos_vision_model import (
            DEFAULT_FPS,
            DEFAULT_MAX_VISION_TOKENS,
            DEFAULT_MIN_VISION_TOKENS,
        )

        validate_config_keys(model_config, self.Cosmos3EdgeHFConfig)
        model_id = resolve_model_id(model_id)
        entry = get_model(model_id)
        if entry is not None and entry.hf_reasoner_class is None:
            raise ValueError(
                f"{model_id} has no in-process Transformers reasoner; use "
                f"Cosmos3ReasonerModel against a vLLM server (just c3-serve-reason model={model_id})."
            )
        self._hf_class = entry.hf_reasoner_class if entry else "Cosmos3EdgeForConditionalGeneration"
        self._min_transformers = entry.min_transformers if entry else "5.19.0"
        self.config = {
            "model_id": model_id,
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "reasoning": False,
            "show_thinking": False,
            "fps": DEFAULT_FPS,
            "min_vision_tokens": DEFAULT_MIN_VISION_TOKENS,
            "max_vision_tokens": DEFAULT_MAX_VISION_TOKENS,
            **model_config,
        }
        self.model = None
        self.processor = None
        logger.debug("config=<%s> | cosmos3-edge hf model (lazy)", self.config)

    # -- loading --------------------------------------------------------------
    @override
    def _load_model(self) -> None:
        import torch
        import transformers

        if self._min_transformers and not _version_at_least(transformers.__version__, self._min_transformers):
            raise ImportError(
                f"Cosmos3-Edge needs transformers>={self._min_transformers} "
                f"(found {transformers.__version__}); it ships models/cosmos3_edge. "
                f"pip install -U 'transformers>={self._min_transformers}'"
            )
        cls = getattr(transformers, self._hf_class, None)
        if cls is None:
            raise ImportError(f"transformers {transformers.__version__} has no {self._hf_class}")

        dtype_name = self.config.get("torch_dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_name in ("auto", "bfloat16") else getattr(torch, dtype_name)
        model_id = self.config["model_id"]
        logger.debug("model_id=<%s> | loading Cosmos3-Edge understanding tower", model_id)
        self.model = cls.from_pretrained(
            model_id, dtype=dtype, device_map=self.config.get("device_map", "auto")
        ).eval()
        # AutoProcessor needs torchvision (Cosmos3EdgeVideoProcessor) even for stills.
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        logger.debug("cosmos3-edge loaded: %s", type(self.model).__name__)

    # -- streaming with <think> routing -------------------------------------
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
        if self.model is None:
            self._load_model()
        parent = super().stream(messages, tool_specs, system_prompt, tool_choice=tool_choice, **kwargs)
        if self.config.get("show_thinking"):
            async for ev in parent:
                yield ev
            return
        async for ev in _route_thinking(parent):
            yield ev


async def _route_thinking(events: AsyncGenerator[StreamEvent, None]) -> AsyncGenerator[StreamEvent, None]:
    """Turn ``<think>…</think>`` text deltas into ``reasoningContent`` deltas.

    Pure stream transform (unit-testable without a model). The Edge chat
    template opens the think block itself, so the model's first tokens are
    thinking *without* an opening tag: text is held back until the first
    ``</think>`` (then emitted as reasoning) or until the stream ends (then
    emitted as plain text, so a non-thinking fine-tune loses nothing). After an
    explicit ``<think>`` the reasoning streams incrementally.
    """
    buf = ""
    state = "head"  # head | think | answer
    tail = len(THINK_CLOSE) - 1

    async for ev in events:
        delta = ev.get("contentBlockDelta", {}).get("delta", {}) if isinstance(ev, dict) else {}
        if "text" not in delta:
            if "contentBlockStop" in ev and buf:
                yield {"contentBlockDelta": {"delta": {"text": buf}}}  # no </think> ever came
                buf = ""
            yield ev
            continue
        buf += delta["text"]
        while buf:
            if state == "answer":
                yield {"contentBlockDelta": {"delta": {"text": buf}}}
                buf = ""
                break
            idx = buf.find(THINK_CLOSE)
            if idx >= 0:
                thinking = buf[:idx]
                if state == "head" and thinking.lstrip().startswith(THINK_OPEN):
                    thinking = thinking.lstrip()[len(THINK_OPEN):]
                if thinking.strip():
                    yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": thinking}}}}
                buf = buf[idx + len(THINK_CLOSE):].lstrip("\n")
                state = "answer"
                continue
            if state == "head" and buf.lstrip().startswith(THINK_OPEN):
                buf = buf.lstrip()[len(THINK_OPEN):]
                state = "think"
                continue
            if state == "think" and len(buf) > tail:
                yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": buf[:-tail]}}}}
                buf = buf[-tail:]
            break  # need more text
    if buf:
        yield {"contentBlockDelta": {"delta": {"text": buf}}}


__all__ = ["Cosmos3EdgeHFModel", "DEFAULT_MODEL"]
