# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos Reason model provider for Strands Agents.

Text-only inference using Cosmos-Reason2 via Hugging Face Transformers.
For vision/video capabilities, use CosmosVisionModel.

- Cosmos-Reason2: https://github.com/nvidia-cosmos/cosmos-reason2
- Models: https://huggingface.co/collections/nvidia/cosmos-reason2
"""

import json
import logging
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*tied weights.*")
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel
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

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


class CosmosModel(Model):
    """NVIDIA Cosmos Reason text model provider.

    Uses Cosmos-Reason2 (based on Qwen3-VL) for text-only inference with
    chain-of-thought reasoning capabilities.

    Example:
        >>> from strands import Agent
        >>> from strands_cosmos import CosmosModel
        >>> model = CosmosModel(model_id="nvidia/Cosmos-Reason2-2B")
        >>> agent = Agent(model=model)
        >>> agent("What are the physics of a ball rolling down a ramp?")
    """

    class CosmosConfig(TypedDict, total=False):
        """Cosmos model configuration."""

        model_id: str
        params: Optional[Dict[str, Any]]
        device_map: str
        torch_dtype: str
        reasoning: bool
        attn_implementation: str

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        **model_config: Unpack[CosmosConfig],
    ) -> None:
        """Initialize Cosmos provider.

        Args:
            model_id: Model identifier (HF Hub ID or local path).
                Supported: nvidia/Cosmos-Reason2-2B, nvidia/Cosmos-Reason2-8B
            **model_config: Configuration options.
        """
        validate_config_keys(model_config, self.CosmosConfig)

        self.config: Dict[str, Any] = {
            "model_id": model_id,
            "device_map": "auto",
            "torch_dtype": "auto",
            "reasoning": False,
            "attn_implementation": "sdpa",
            **model_config,
        }

        logger.debug("config=<%s> | initializing cosmos model", self.config)
        self._load_model()

    def _load_model(self) -> None:
        """Load model using transformers."""
        import torch
        import transformers

        model_id = self.config["model_id"]
        dtype = self.config["torch_dtype"]
        if dtype == "auto":
            dtype = torch.float16

        logger.debug("model_id=<%s> | loading", model_id)

        # Suppress noisy "tied weights" message from transformers
        import logging as _logging
        _hf_logger = _logging.getLogger("transformers.modeling_utils")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)

        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=self.config["device_map"],
            attn_implementation=self.config.get("attn_implementation", "sdpa"),
        )

        _hf_logger.setLevel(_prev_level)
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_id)

        logger.debug("cosmos model loaded")

    @override
    def update_config(self, **model_config: Unpack[CosmosConfig]) -> None:  # type: ignore[override]
        """Update configuration."""
        validate_config_keys(model_config, self.CosmosConfig)

        if "model_id" in model_config and model_config["model_id"] != self.config.get("model_id"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> CosmosConfig:
        """Get configuration."""
        return self.config  # type: ignore[return-value]

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> Dict[str, Any]:
        """Format a content block."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        if "image" in content:
            return {"type": "text", "text": "[Image content - use CosmosVisionModel]"}
        if "document" in content:
            return {"type": "text", "text": "[Document content not supported]"}
        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> Dict[str, Any]:
        """Format a tool call."""
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
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

    @classmethod
    def format_request_messages(
        cls, messages: Messages, system_prompt: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Format messages array for Cosmos."""
        formatted_messages: list[Dict[str, Any]] = []
        if system_prompt:
            formatted_messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(
                    block_type in content
                    for block_type in ["toolResult", "toolUse", "reasoningContent"]
                )
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            text_content = " ".join(c["text"] for c in formatted_contents if c["type"] == "text")

            formatted_message = {
                "role": message["role"],
                "content": text_content if text_content else "",
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [m for m in formatted_messages if m.get("content") or "tool_calls" in m]

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format request."""
        formatted_messages = self.format_request_messages(messages, system_prompt)

        tools = None
        if tool_specs:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": ts["name"],
                        "description": ts["description"],
                        "parameters": ts["inputSchema"]["json"],
                    },
                }
                for ts in tool_specs
            ]

        return {"messages": formatted_messages, "tools": tools}

    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format event into StreamEvent."""
        chunk_type = event["chunk_type"]

        if chunk_type == "message_start":
            return {"messageStart": {"role": "assistant"}}
        if chunk_type == "content_start":
            if event.get("data_type") == "tool":
                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": event["data"]["name"],
                                "toolUseId": event["data"]["id"],
                            }
                        }
                    }
                }
            return {"contentBlockStart": {"start": {}}}
        if chunk_type == "content_delta":
            data_type = event.get("data_type", "text")
            if data_type == "reasoning_content":
                return {
                    "contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}
                }
            if data_type == "tool":
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"]}}}}
            return {"contentBlockDelta": {"delta": {"text": event["data"]}}}
        if chunk_type == "content_stop":
            return {"contentBlockStop": {}}
        if chunk_type == "message_stop":
            reason = event.get("data", "end_turn")
            if reason == "tool_calls":
                return {"messageStop": {"stopReason": "tool_use"}}
            if reason == "length":
                return {"messageStop": {"stopReason": "max_tokens"}}
            return {"messageStop": {"stopReason": "end_turn"}}
        if chunk_type == "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"]["input_tokens"],
                        "outputTokens": event["data"]["output_tokens"],
                        "totalTokens": event["data"]["input_tokens"]
                        + event["data"]["output_tokens"],
                    },
                    "metrics": {"latencyMs": 0},
                },
            }

        raise RuntimeError(f"chunk_type=<{chunk_type}> | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation using Cosmos-Reason2.

        Args:
            messages: List of message objects.
            tool_specs: List of tool specifications.
            system_prompt: System prompt.
            tool_choice: Tool choice selection.
            **kwargs: Additional arguments.

        Yields:
            Formatted message chunks.
        """
        import transformers

        warn_on_tool_choice_not_supported(tool_choice)

        request = self.format_request(messages, tool_specs, system_prompt)

        # Add reasoning prompt if enabled
        if self.config.get("reasoning"):
            msgs = request["messages"]
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += f"\n\n{REASONING_PROMPT}"

        # Apply chat template
        try:
            if request["tools"]:
                text = self.processor.apply_chat_template(
                    request["messages"],
                    tools=request["tools"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = self.processor.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            logger.warning(f"tools not supported by template, falling back: {e}")
            text = self.processor.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )

        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 4096)

        # Tokenize
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        # Use streamer for token-by-token generation
        streamer = transformers.TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        import threading

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
        }
        # Add sampling params
        if "temperature" in params:
            gen_kwargs["temperature"] = params["temperature"]
            gen_kwargs["do_sample"] = params["temperature"] > 0
        if "top_p" in params:
            gen_kwargs["top_p"] = params["top_p"]

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream tokens
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        token_count = 0
        full_text = ""
        tool_calls: list[Dict[str, Any]] = []

        for text_chunk in streamer:
            if text_chunk:
                full_text += text_chunk
                yield self.format_chunk(
                    {"chunk_type": "content_delta", "data_type": "text", "data": text_chunk}
                )
                token_count += 1

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        # Check for tool calls in response
        finish_reason = "end_turn"
        try:
            # Some models emit JSON tool calls
            if '{"name":' in full_text and '"arguments":' in full_text:
                import re

                tool_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
                matches = re.findall(tool_pattern, full_text)
                for name, args_str in matches:
                    tool_calls.append({"name": name, "arguments": json.loads(args_str)})
                    finish_reason = "tool_calls"
        except Exception:
            pass

        # Emit tool calls
        for i, tc in enumerate(tool_calls):
            tool_id = f"{tc['name']}_{i}"
            yield self.format_chunk(
                {
                    "chunk_type": "content_start",
                    "data_type": "tool",
                    "data": {"name": tc["name"], "id": tool_id},
                }
            )
            yield self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "tool",
                    "data": json.dumps(tc.get("arguments", {})),
                }
            )
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        # Metadata
        input_tokens = len(self.processor.tokenizer.encode(text))
        yield self.format_chunk(
            {
                "chunk_type": "metadata",
                "data": {"input_tokens": input_tokens, "output_tokens": token_count},
            }
        )

        thread.join()

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """Get structured output with JSON schema."""
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON:\n{json.dumps(schema, indent=2)}"
        augmented_system_prompt = (system_prompt or "") + json_instruction

        response_text = ""
        async for event in self.stream(prompt, system_prompt=augmented_system_prompt, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            yield cast(Dict[str, Union[T, Any]], event)

        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            data = json.loads(response_text.strip())
            yield {"output": output_model(**data)}
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {e}\nResponse: {response_text}") from e
