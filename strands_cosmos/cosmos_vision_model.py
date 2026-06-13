# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos Reason Vision model provider for Strands Agents.

Multimodal inference (video + image + text) using Cosmos-Reason2 via Transformers.
Specialized for physical AI reasoning, driving analysis, robot planning, and video understanding.

- Cosmos-Reason2: https://github.com/nvidia-cosmos/cosmos-reason2
- Models: https://huggingface.co/collections/nvidia/cosmos-reason2
"""

import json
import logging
import re
import threading
import warnings

# Suppress noisy video decoding warnings from transformers/torchvision
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchvision.*decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*tied weights.*")
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from strands.models._validation import (
    validate_config_keys,
    warn_on_tool_choice_not_supported,
)
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from typing_extensions import TypedDict, Unpack, override

logger = logging.getLogger(__name__)


def _resolve_media_ref(ref: str):
    """Validate an ``<image>``/``<video>`` media reference from the model.

    Returns a safe path/URL string, or ``None`` to drop an invalid reference.
    Local paths are confined to the project workspace and remote URLs are checked
    against the allowed-host policy before they reach the media pipeline.
    """
    if not ref:
        return None
    try:
        from .tools._security import resolve_in_workspace, validate_url
        if ref.startswith(("http://", "https://")):
            return validate_url(ref, allow_public=True)
        return str(resolve_in_workspace(ref, must_exist=True))
    except Exception as _e:  # SecurityError or anything unexpected -> drop
        logger.warning("dropping unsafe media reference %r: %s", ref, _e)
        return None

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"

# Cosmos-Reason2 vision constants
PIXELS_PER_TOKEN = 32**2  # 1024 pixels per visual token
DEFAULT_FPS = 4
DEFAULT_MAX_VISION_TOKENS = 8192
DEFAULT_MIN_VISION_TOKENS = 256

REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""

# Built-in task prompts from Cosmos-Reason2
TASK_PROMPTS = {
    "caption": "Caption the video in detail.",
    "embodied_reasoning": "What can be the next immediate action?",
    "driving": "The video depicts the observation from the vehicle's camera. You need to think step by step and identify the objects in the scene that are critical for safe navigation.",
    "causal": "What will the person likely do next in this situation?",
    "temporal_localization": "Describe the notable events in the provided video.",
    "2d_grounding": "Locate the bounding box of {object_name}. Return a json.",
    "robot_cot": 'You are given the task "{task_instruction}". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {{"point_2d": [x, y], "label": "gripper trajectory"}}.',
    "describe_anything": 'Please caption the notable attributes in the provided image. List and describe all marked subjects in the image with their categories and detailed captions using a json with keyword "subject_id", "category" and "caption".',
    "mvp_bench": "Is this video physically plausible/possible according to your understanding of e.g. object permanence, shape constancy (objects maintain shape over time), continuous trajectories of objects?",
}


class CosmosVisionModel(Model):
    """NVIDIA Cosmos Reason Vision model provider.

    Multimodal model for video understanding, physical AI reasoning,
    driving analysis, and embodied intelligence.

    Example:
        >>> from strands import Agent
        >>> from strands_cosmos import CosmosVisionModel
        >>> model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
        >>> agent = Agent(model=model)
        >>> agent("Describe: <video>dashcam.mp4</video>")
        >>> agent("What happens next: <image>scene.jpg</image>")
    """

    class CosmosVisionConfig(TypedDict, total=False):
        """Cosmos Vision model configuration."""

        model_id: str
        params: Optional[Dict[str, Any]]
        device_map: str
        torch_dtype: str
        reasoning: bool
        fps: float
        min_vision_tokens: int
        max_vision_tokens: int

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        **model_config: Unpack[CosmosVisionConfig],
    ) -> None:
        """Initialize Cosmos Vision provider.

        Args:
            model_id: Model identifier (HF Hub ID or local path).
            **model_config: Configuration options.
        """
        validate_config_keys(model_config, self.CosmosVisionConfig)

        self.config: Dict[str, Any] = {
            "model_id": model_id,
            "device_map": "auto",
            "torch_dtype": "auto",
            "reasoning": False,
            "fps": DEFAULT_FPS,
            "min_vision_tokens": DEFAULT_MIN_VISION_TOKENS,
            "max_vision_tokens": DEFAULT_MAX_VISION_TOKENS,
            **model_config,
        }

        logger.debug("config=<%s> | initializing cosmos vision model", self.config)
        self._load_model()

    def _load_model(self) -> None:
        """Load model using transformers."""
        import torch
        import transformers

        model_id = self.config["model_id"]
        dtype = self.config["torch_dtype"]
        if dtype == "auto":
            dtype = torch.float16

        logger.debug("model_id=<%s> | loading vision model", model_id)

        # Suppress noisy "tied weights" message from transformers
        import logging as _logging
        _hf_logger = _logging.getLogger("transformers.modeling_utils")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)

        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=self.config["device_map"],
            attn_implementation="sdpa",
        )

        _hf_logger.setLevel(_prev_level)
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_id)

        # Configure vision token limits
        min_tokens = self.config["min_vision_tokens"]
        max_tokens = self.config["max_vision_tokens"]
        self.processor.image_processor.size = {
            "shortest_edge": min_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_tokens * PIXELS_PER_TOKEN,
        }
        self.processor.video_processor.size = {
            "shortest_edge": min_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_tokens * PIXELS_PER_TOKEN,
        }

        logger.debug("cosmos vision model loaded")

    @override
    def update_config(self, **model_config: Unpack[CosmosVisionConfig]) -> None:  # type: ignore[override]
        """Update configuration."""
        validate_config_keys(model_config, self.CosmosVisionConfig)

        if "model_id" in model_config and model_config["model_id"] != self.config.get("model_id"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> CosmosVisionConfig:
        """Get configuration."""
        return self.config  # type: ignore[return-value]

    def _extract_media_from_messages(
        self, messages: Messages
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Extract images, videos from Strands messages and build Cosmos conversation.

        Args:
            messages: Strands message format.

        Returns:
            Tuple of (conversation_messages, image_paths, video_paths)
        """
        images: List[str] = []
        videos: List[str] = []
        chat_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            if role == "system":
                continue

            user_content: List[Dict[str, Any]] = []
            text_parts: List[str] = []

            for content in message["content"]:
                if "image" in content:
                    img_data = content["image"]
                    if "format" in img_data and "source" in img_data:
                        import tempfile

                        source = img_data["source"]
                        if "bytes" in source:
                            # Save to temp file for Cosmos
                            fmt = img_data.get("format", "jpeg")
                            tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
                            tmp.write(source["bytes"])
                            tmp.close()
                            images.append(tmp.name)
                            user_content.append({"type": "image", "image": tmp.name})
                    elif "source" in img_data:
                        source = img_data["source"]
                        if source.get("type") == "url":
                            images.append(source["url"])
                            user_content.append({"type": "image", "image": source["url"]})

                elif "text" in content:
                    text_content = content["text"]

                    # Extract <image>path</image> tags. Paths are LLM-controlled,
                    # so confine each to the workspace allow-list before handing it
                    # to the HF media-loading pipeline (CWE-22). Remote URLs are
                    # validated against the SSRF policy.
                    for img_path in re.findall(r"<image>(.*?)</image>", text_content):
                        resolved = _resolve_media_ref(img_path.strip())
                        if resolved is None:
                            continue
                        images.append(resolved)
                        user_content.append({"type": "image", "image": resolved})
                    text_content = re.sub(r"<image>.*?</image>", "", text_content)

                    # Extract <video>path</video> tags (same containment).
                    for vid_path in re.findall(r"<video>(.*?)</video>", text_content):
                        resolved = _resolve_media_ref(vid_path.strip())
                        if resolved is None:
                            continue
                        videos.append(resolved)
                        user_content.append({"type": "video", "video": resolved})
                    text_content = re.sub(r"<video>.*?</video>", "", text_content)

                    cleaned = text_content.strip()
                    if cleaned:
                        text_parts.append(cleaned)

            # Build text prompt
            if text_parts:
                user_content.append({"type": "text", "text": " ".join(text_parts)})
            elif images or videos:
                user_content.append({"type": "text", "text": "Describe what you see."})

            if user_content:
                chat_messages.append({"role": role, "content": user_content})

        return chat_messages, images, videos

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
        """Stream generation with video + image support.

        Supports inline media tags in user messages:
        - <video>path/to/video.mp4</video>
        - <image>path/to/image.jpg</image>

        Args:
            messages: Conversation messages with optional media.
            tool_specs: Tool specifications (limited support for vision).
            system_prompt: System prompt.
            tool_choice: Tool selection strategy.
            **kwargs: Additional arguments.

        Yields:
            StreamEvent dictionaries.
        """
        import transformers

        if tool_choice:
            warn_on_tool_choice_not_supported(tool_choice)

        # Extract media
        chat_messages, images, videos = self._extract_media_from_messages(messages)

        # Add system prompt
        if system_prompt:
            chat_messages.insert(
                0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        # Add reasoning if enabled
        if self.config.get("reasoning") and chat_messages:
            last_msg = chat_messages[-1]
            if last_msg["role"] == "user":
                for item in last_msg["content"]:
                    if item.get("type") == "text":
                        item["text"] += f"\n\n{REASONING_PROMPT}"
                        break

        # Process with Cosmos processor
        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 4096)
        fps = self.config.get("fps", DEFAULT_FPS)

        inputs = self.processor.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )
        inputs = inputs.to(self.model.device)

        # Use streamer
        streamer = transformers.TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
        }
        if "temperature" in params:
            gen_kwargs["temperature"] = params["temperature"]
            gen_kwargs["do_sample"] = params["temperature"] > 0
        if "top_p" in params:
            gen_kwargs["top_p"] = params["top_p"]

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream response
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        token_count = 0
        for text_chunk in streamer:
            if text_chunk:
                yield {"contentBlockDelta": {"delta": {"text": text_chunk}}}
                token_count += 1

        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

        # Metadata
        input_token_count = inputs["input_ids"].shape[-1] if hasattr(inputs, "__getitem__") else 0
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": input_token_count,
                    "outputTokens": token_count,
                    "totalTokens": input_token_count + token_count,
                },
                "metrics": {"latencyMs": 0},
            }
        }

        thread.join()

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> Dict[str, Any]:
        """Format a content block."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        if "image" in content:
            return {"type": "text", "text": "[Image]"}
        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> Dict[str, Any]:
        """Format a tool call."""
        return {
            "function": {"arguments": json.dumps(tool_use["input"]), "name": tool_use["name"]},
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> Dict[str, Any]:
        """Format a tool result message."""
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )
        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [cls.format_request_message_content(content) for content in contents],
        }

    @override
    async def structured_output(
        self,
        output_model: Any,
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get structured output (limited support for vision models)."""
        logger.warning("structured_output has limited support for vision models")
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON:\n{json.dumps(schema, indent=2)}"
        augmented_system_prompt = (system_prompt or "") + json_instruction

        response_text = ""
        async for event in self.stream(prompt, system_prompt=augmented_system_prompt, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            yield event

        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            data = json.loads(response_text.strip())
            yield {"output": output_model(**data)}
        except Exception as e:
            raise ValueError(f"Failed to parse: {e}\nResponse: {response_text}") from e
