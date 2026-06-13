# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Reason2 inference via HuggingFace Transformers (full-precision)."""
from __future__ import annotations

from pathlib import Path

from strands import tool
from ._common import ok, err


@tool
def cosmos_reason_hf(
    prompt: str,
    image_path: str = "",
    video_path: str = "",
    model_id: str = "nvidia/Cosmos-Reason2-2B",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    device: str = "auto",
    return_image: bool = False,
) -> dict:
    """Run Cosmos-Reason2 inference directly via HuggingFace Transformers.

    Useful on x86 GPU host for full-precision reference outputs, or when
    the TRT-EdgeLLM server is not available. For Thor edge inference,
    use `cosmos_inference` (TRT server) instead.

    Args:
        prompt: User instruction.
        image_path: Path to image (for VLM). Mutually exclusive with video_path.
        video_path: Path to video (for VQA). Frames auto-sampled.
        model_id: HF model id (default nvidia/Cosmos-Reason2-2B).
        max_new_tokens: Token cap.
        temperature: Sampling temperature.
        device: "auto" | "cuda" | "cpu".
        return_image: If True, return the input image in the response.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries the model's text response; on error ``status`` is ``"error"`` with a message.
    """
    if not image_path and not video_path:
        return err("image_path or video_path is required")
    if image_path and video_path:
        return err("provide exactly one of image_path or video_path")

    try:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        return err(f"missing deps: {e}. pip install transformers torch qwen-vl-utils")

    try:
        proc = transformers.Qwen3VLProcessor.from_pretrained(model_id)
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device,
            attn_implementation="sdpa",
        )

        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path:
            p = Path(image_path).expanduser()
            if not p.exists():
                return err(f"image not found: {p}")
            content.insert(0, {"type": "image", "image": str(p)})
        else:
            p = Path(video_path).expanduser()
            if not p.exists():
                return err(f"video not found: {p}")
            content.insert(0, {"type": "video", "video": str(p)})

        messages = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        output = proc.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        image_bytes = None
        image_format = None
        if return_image and image_path:
            image_bytes = Path(image_path).expanduser().read_bytes()
            image_format = "jpeg"

        return ok(
            text=f"Cosmos-Reason2 HF → {output}",
            data={"model_id": model_id, "output_chars": len(output)},
            image_bytes=image_bytes,
            image_format=image_format,
        )
    except Exception as e:
        return err(f"HF inference failed: {type(e).__name__}: {e}")
