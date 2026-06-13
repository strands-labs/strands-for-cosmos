# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Cosmos 3 Generator model provider for Strands Agents.

The Cosmos 3 **Generator** surface produces vision/sound/action from text,
image, video, and action inputs. This provider wraps the in-process HuggingFace
Diffusers `Cosmos3OmniPipeline` for local generation (no NIM).

Supported modes (Diffusers):
    - text2image   : text → PIL image (num_frames=1)
    - text2video   : text → video frames (default 189f @ 24fps ≈ 7.9s)
    - image2video  : text + image → video frames
    - text2video-with-sound : text → video + audio (sound-capable checkpoints)

Because the Strands `Model` interface is text-streaming-centric, generated media
is written to disk and the provider streams back a short status + output path.
For programmatic generation prefer the `cosmos3_*` tools (justfile-backed).

    >>> from strands_cosmos import Cosmos3GeneratorModel
    >>> m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")
    >>> out = m.generate(mode="text2video", prompt="A robot in a warehouse.",
    ...                   out_path="/tmp/c3.mp4", num_frames=49, num_inference_steps=15)

- Diffusers docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3
- Models: https://huggingface.co/nvidia/Cosmos3-Nano
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    cast,
)

from strands.models._validation import validate_config_keys
from strands.models.model import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import TypedDict, Unpack, override

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"

# Generator prompt-upsampling / sampling defaults from the Cosmos 3 README.
GEN_DEFAULTS = {
    "num_frames": 189,
    "fps": 24,
    "height": 480,
    "width": 832,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "flow_shift": 10.0,
    "seed": 0,
}

# Resolution tier → (width, height) for 16:9.
RES_TIERS = {
    "256": (320, 192),
    "480": (832, 480),
    "720": (1280, 720),
}


class Cosmos3GeneratorModel(Model):
    """NVIDIA Cosmos 3 Generator provider (Diffusers backend, in-process).

    Generates images / videos (optionally with sound) from text and image inputs
    using `Cosmos3OmniPipeline`. Lazy-loads the pipeline on first generation.
    """

    class Cosmos3GeneratorConfig(TypedDict, total=False):
        """Cosmos 3 Generator configuration."""

        model_id: str
        torch_dtype: str
        device_map: str
        flow_shift: float
        guardrails: bool
        params: Optional[Dict[str, Any]]

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        **model_config: Unpack[Cosmos3GeneratorConfig],
    ) -> None:
        """Initialize the Cosmos 3 Generator provider.

        Args:
            model_id: HF model id (default nvidia/Cosmos3-Nano).
            **model_config: dtype/device/guardrails/flow_shift overrides.
        """
        validate_config_keys(model_config, self.Cosmos3GeneratorConfig)

        self.config: Dict[str, Any] = {
            "model_id": model_id,
            "torch_dtype": "bfloat16",
            "device_map": "cuda",
            "flow_shift": GEN_DEFAULTS["flow_shift"],
            "guardrails": True,
            "params": {},
        }
        self.config.update(model_config)
        logger.debug("config=<%s> | initializing cosmos3 generator", self.config)

        self._pipe = None
        self._lock = threading.Lock()

    # ── Strands Model API ─────────────────────────────────────────────────
    @override
    def update_config(self, **model_config: Unpack[Cosmos3GeneratorConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.Cosmos3GeneratorConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> Cosmos3GeneratorConfig:  # type: ignore[override]
        return cast(Cosmos3GeneratorModel.Cosmos3GeneratorConfig, self.config)

    def _load_pipeline(self):
        """Lazy-load Cosmos3OmniPipeline (thread-safe, once)."""
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                import torch
                from diffusers import Cosmos3OmniPipeline
                from diffusers.schedulers.scheduling_unipc_multistep import (
                    UniPCMultistepScheduler,
                )
            except ImportError as e:
                raise ImportError(
                    "Cosmos3GeneratorModel needs diffusers(main)+torch. "
                    "Run `just c3-setup-gen`. Original error: " + str(e)
                ) from e

            dtype = getattr(torch, self.config["torch_dtype"], torch.bfloat16)
            logger.info("loading Cosmos3OmniPipeline model_id=%s", self.config["model_id"])
            pipe = Cosmos3OmniPipeline.from_pretrained(
                self.config["model_id"],
                torch_dtype=dtype,
                device_map=self.config["device_map"],
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=self.config["flow_shift"]
            )
            self._pipe = pipe
            return self._pipe

    def generate(
        self,
        mode: str = "text2video",
        prompt: str = "",
        out_path: str = "/tmp/cosmos3_out.mp4",
        image: Optional[str] = None,
        negative_prompt: str = "",
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        resolution: str = "480",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        enable_sound: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a Cosmos 3 generation and write output to disk.

        Args:
            mode: text2image | text2video | image2video | text2video-with-sound
            prompt: positive text prompt.
            out_path: output file path (.png for image, .mp4 for video).
            image: input image path for image2video.
            negative_prompt: concepts to avoid.
            num_frames: frame count (1 for image).
            fps: frames per second.
            resolution: "256" | "480" | "720".
            num_inference_steps: diffusion steps.
            guidance_scale: CFG scale.
            enable_sound: produce audio (sound-capable modes).
            seed: reproducibility seed.

        Returns:
            Dict with status, mode, out_path, and params used.
        """
        import torch
        from diffusers.utils import export_to_video

        pipe = self._load_pipeline()

        width, height = RES_TIERS.get(resolution, RES_TIERS["720"])
        if mode == "text2image":
            num_frames = 1

        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames if num_frames is not None else GEN_DEFAULTS["num_frames"],
            "height": height,
            "width": width,
            "fps": fps if fps is not None else GEN_DEFAULTS["fps"],
            "num_inference_steps": (
                num_inference_steps if num_inference_steps is not None
                else GEN_DEFAULTS["num_inference_steps"]
            ),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None
                else GEN_DEFAULTS["guidance_scale"]
            ),
            "enable_sound": enable_sound,
            "add_resolution_template": False,
            "add_duration_template": False,
            "generator": torch.Generator(device="cuda").manual_seed(
                seed if seed is not None else GEN_DEFAULTS["seed"]
            ),
        }

        if image and mode in ("image2video", "image2video-with-sound"):
            from diffusers.utils import load_image
            params["image"] = load_image(image)
        else:
            params["image"] = None

        params.update(self.config.get("params") or {})
        params.update(kwargs)

        logger.info("cosmos3 generate mode=%s steps=%s frames=%s",
                    mode, params["num_inference_steps"], params["num_frames"])
        result = pipe(**params)

        # Confine the (caller/LLM-supplied) output path to the workspace
        # allow-list before creating dirs / writing bytes (CWE-22).
        from .tools._security import resolve_output_path
        out_path = str(resolve_output_path(out_path))

        # Pull video frames + optional sound from the omni pipeline output.
        video_frames = getattr(result, "video", None)
        if video_frames is None and hasattr(result, "images"):
            video_frames = result.images
        sound = getattr(result, "sound", None)

        has_audio = False
        if mode == "text2image":
            img = video_frames[0] if isinstance(video_frames, list) else result.images[0]
            if not out_path.lower().endswith((".png", ".jpg", ".jpeg")):
                out_path = os.path.splitext(out_path)[0] + ".png"
            img.save(out_path)
        else:
            if enable_sound and sound is not None:
                # Write silent video to temp, mux stereo audio via ffmpeg.
                tmp_video = out_path + ".video.mp4"
                export_to_video(video_frames, tmp_video, fps=params["fps"], macro_block_size=1)
                has_audio = self._mux_audio(tmp_video, sound, out_path, sample_rate=48000)
                try:
                    os.remove(tmp_video)
                except OSError:
                    pass
                if not has_audio:
                    # Fallback: video only
                    export_to_video(video_frames, out_path, fps=params["fps"], macro_block_size=1)
            else:
                export_to_video(video_frames, out_path, fps=params["fps"], macro_block_size=1)

        return {
            "status": "success",
            "mode": mode,
            "out_path": out_path,
            "has_audio": has_audio,
            "params": {k: v for k, v in params.items() if k not in ("generator", "image")},
        }

    @staticmethod
    def _mux_audio(video_path: str, sound, out_path: str, sample_rate: int = 48000) -> bool:
        """Mux a stereo audio tensor into a video via ffmpeg. Returns True on success."""
        import shutil
        import subprocess

        import soundfile as sf

        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found; cannot mux audio")
            return False
        try:
            arr = sound.detach().to("cpu", dtype=__import__("torch").float32).numpy()
            # Expect shape (channels, samples) or (samples, channels); normalize to (samples, channels)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.shape[0] in (1, 2) and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            wav_path = out_path + ".wav"
            sf.write(wav_path, arr, sample_rate)
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", wav_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", out_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            try:
                os.remove(wav_path)
            except OSError:
                pass
            if r.returncode != 0:
                logger.warning("ffmpeg mux failed: %s", r.stderr[-300:])
                return False
            return True
        except Exception as e:
            logger.warning("audio mux error: %s", e)
            return False

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
        """Treat the last user text as a text2video prompt; stream a status report.

        For full control (modes, sound, image conditioning) use the
        `cosmos3_*` generator tools instead.
        """
        # Extract the last user text as the prompt
        prompt = ""
        for message in reversed(messages):
            if message["role"] == "user":
                for content in message["content"]:
                    if "text" in content:
                        prompt = content["text"]
                        break
                if prompt:
                    break

        out_path = kwargs.get("out_path", "/tmp/cosmos3_t2v.mp4")
        mode = kwargs.get("mode", "text2video")

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"Generating ({mode})... this can take a while.\n"}}}

        try:
            result = self.generate(mode=mode, prompt=prompt, out_path=out_path, **{
                k: v for k, v in kwargs.items() if k not in ("out_path", "mode")
            })
            text = f"Done. Wrote {result['out_path']}\nParams: {json.dumps(result['params'])}"
        except Exception as e:
            logger.error("cosmos3 generator error: %s", e)
            text = f"[Cosmos3 generator error: {e}]"

        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
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
        """Not supported for generator models."""
        raise NotImplementedError("structured_output is not supported for Cosmos3GeneratorModel")
