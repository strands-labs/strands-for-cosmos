# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strands Cosmos - NVIDIA Cosmos ecosystem provider for Strands Agents.

Full-lifecycle support across Cosmos 3 (omnimodal) and Cosmos-Reason2 (VLM):
reasoning, world-model generation (image/video/audio/action), Predict2.5,
Transfer2.5, Xenna curation, quantization, edge deployment, and evaluation.

Model Providers (use as Agent model):
  - Cosmos3ReasonerModel: Cosmos 3 omnimodal reasoning (text+vision -> text) via vLLM
  - Cosmos3GeneratorModel: Cosmos 3 generation (-> image/video/sound) via Diffusers
  - CosmosVisionModel: Reason2 VLM (video + image + text) via HF Transformers
  - CosmosModel: Reason2 text-only via HF Transformers

Tools (use inside any Agent):
  - 49 tools covering the full Cosmos pipeline via justfile recipes
    (incl. 28 cosmos3_* tools: reason / generate (incl. sound) / action /
     prompt-upsampling / batch-caption / videophy2-eval / training / serve)
  - See strands_cosmos.tools for the complete list
"""

try:
    from strands_cosmos._version import version as __version__
except ImportError:  # not installed via setuptools-scm build (e.g. editable pre-tag)
    __version__ = "0.0.0+unknown"


from strands_cosmos.cosmos3_generator_model import Cosmos3GeneratorModel
from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel
from strands_cosmos.cosmos_model import CosmosModel
from strands_cosmos.cosmos_vision_model import CosmosVisionModel

# Export all tools for convenient access
from strands_cosmos.tools import (
    cosmos3_action_cot,
    cosmos3_caption,
    cosmos3_caption_batch,
    cosmos3_embodied,
    cosmos3_eval_videophy2,
    cosmos3_forward_dynamics,
    cosmos3_ground,
    cosmos3_image2video,
    cosmos3_image2video_sound,
    cosmos3_inverse_dynamics,
    cosmos3_plausibility,
    cosmos3_policy,
    # Cosmos 3
    cosmos3_reason,
    cosmos3_serve,
    cosmos3_situation,
    cosmos3_temporal,
    cosmos3_text2image,
    cosmos3_text2video,
    cosmos3_text2video_sound,
    cosmos3_train,
    cosmos3_train_convert,
    cosmos3_train_convert_vlm,
    cosmos3_train_export,
    cosmos3_train_prep_dataset,
    # Cosmos 3 training
    cosmos3_train_recipes,
    cosmos3_train_show,
    cosmos3_upsample_prompt,
    cosmos3_video2video,
    cosmos_build_engine,
    # Data
    cosmos_curate,
    cosmos_distill,
    # Evaluation
    cosmos_evaluate,
    cosmos_export_onnx,
    # Reason2 VLM
    cosmos_inference,
    # Legacy
    cosmos_invoke,
    # Model lifecycle
    cosmos_model_download,
    # Training
    cosmos_post_train,
    # Predict2.5
    cosmos_predict_generate,
    cosmos_quantize,
    cosmos_reason_hf,
    cosmos_serve,
    # System
    cosmos_sysinfo,
    # Transfer2.5
    cosmos_transfer_generate,
    cosmos_vision_invoke,
    image_read,
    nats_publish,
    # I/O
    rtp_capture_frame,
    video_extract_frames,
    video_probe,
)

__all__ = [
    "__version__",
    # Model providers
    "CosmosModel",
    "CosmosVisionModel",
    "Cosmos3ReasonerModel",
    "Cosmos3GeneratorModel",
    # Reason2 VLM
    "cosmos_inference",
    "cosmos_reason_hf",
    "cosmos_serve",
    # Predict2.5
    "cosmos_predict_generate",
    # Transfer2.5
    "cosmos_transfer_generate",
    # Model lifecycle
    "cosmos_model_download",
    "cosmos_quantize",
    "cosmos_export_onnx",
    "cosmos_build_engine",
    # Training
    "cosmos_post_train",
    "cosmos_distill",
    # Data
    "cosmos_curate",
    # Evaluation
    "cosmos_evaluate",
    # I/O
    "rtp_capture_frame",
    "nats_publish",
    "video_probe",
    "video_extract_frames",
    "image_read",
    # System
    "cosmos_sysinfo",
    "cosmos_invoke",
    "cosmos_vision_invoke",
    # Cosmos 3
    "cosmos3_reason",
    "cosmos3_caption",
    "cosmos3_temporal",
    "cosmos3_embodied",
    "cosmos3_ground",
    "cosmos3_plausibility",
    "cosmos3_situation",
    "cosmos3_action_cot",
    "cosmos3_text2image",
    "cosmos3_text2video",
    "cosmos3_image2video",
    "cosmos3_video2video",
    "cosmos3_text2video_sound",
    "cosmos3_image2video_sound",
    "cosmos3_forward_dynamics",
    "cosmos3_inverse_dynamics",
    "cosmos3_policy",
    "cosmos3_serve",
    # Cosmos 3 training
    "cosmos3_train_recipes",
    "cosmos3_train_show",
    "cosmos3_train_convert",
    "cosmos3_train_convert_vlm",
    "cosmos3_train_prep_dataset",
    "cosmos3_train",
    "cosmos3_train_export",
    "cosmos3_upsample_prompt",
    "cosmos3_caption_batch",
    "cosmos3_eval_videophy2",
]
