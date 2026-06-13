# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strands Cosmos tools — full Cosmos ecosystem coverage via justfile recipes.

Tools are thin Python wrappers over `just <recipe>` commands.
The justfile is the single source of truth for all pipeline commands.

Tool families:
  - Reason2 (VLM): cosmos_inference, cosmos_reason_hf, cosmos_serve
  - Predict2.5 (world model): cosmos_predict_generate
  - Transfer2.5 (ControlNet): cosmos_transfer_generate
  - Model lifecycle: cosmos_model_download, cosmos_quantize, cosmos_export_onnx, cosmos_build_engine
  - Training: cosmos_post_train, cosmos_distill
  - Data: cosmos_curate
  - Evaluation: cosmos_evaluate
  - I/O: rtp_capture_frame, nats_publish, video_probe, video_extract_frames, image_read
  - System: cosmos_sysinfo
  - Legacy (direct HF): cosmos_invoke, cosmos_vision_invoke
"""

# Reason2 VLM
from strands_cosmos.tools.build_engine import cosmos_build_engine

# Cosmos 3 (omnimodal: Reasoner + Generator + Action) ───────────────────
from strands_cosmos.tools.cosmos3 import (
    cosmos3_action_cot,
    cosmos3_caption,
    cosmos3_embodied,
    cosmos3_forward_dynamics,
    cosmos3_ground,
    cosmos3_image2video,
    cosmos3_image2video_sound,
    cosmos3_inverse_dynamics,
    cosmos3_plausibility,
    cosmos3_policy,
    cosmos3_reason,
    cosmos3_serve,
    cosmos3_situation,
    cosmos3_temporal,
    cosmos3_text2image,
    cosmos3_text2video,
    cosmos3_text2video_sound,
    cosmos3_video2video,
)

# Cosmos 3 extra (prompt upsampling / batch captioning / videophy2 eval)
from strands_cosmos.tools.cosmos3_extra import (
    cosmos3_caption_batch,
    cosmos3_eval_videophy2,
    cosmos3_upsample_prompt,
)

# Cosmos 3 post-training (SFT) ──────────────────────────
from strands_cosmos.tools.cosmos3_train import (
    cosmos3_train,
    cosmos3_train_convert,
    cosmos3_train_convert_vlm,
    cosmos3_train_export,
    cosmos3_train_prep_dataset,
    cosmos3_train_recipes,
    cosmos3_train_show,
)

# Legacy (direct HF inference, kept for backward compat) ───────────────
from strands_cosmos.tools.cosmos_invoke import cosmos_invoke
from strands_cosmos.tools.cosmos_vision_invoke import cosmos_vision_invoke

# Data curation
from strands_cosmos.tools.curate import cosmos_curate
from strands_cosmos.tools.distill import cosmos_distill

# Evaluation
from strands_cosmos.tools.evaluate import cosmos_evaluate
from strands_cosmos.tools.export_onnx import cosmos_export_onnx
from strands_cosmos.tools.image_read import image_read
from strands_cosmos.tools.inference import cosmos_inference

# Model lifecycle
from strands_cosmos.tools.model_download import cosmos_model_download
from strands_cosmos.tools.nats_pub import nats_publish

# Training
from strands_cosmos.tools.post_train import cosmos_post_train

# Predict2.5 (world model)
from strands_cosmos.tools.predict_generate import cosmos_predict_generate
from strands_cosmos.tools.quantize import cosmos_quantize
from strands_cosmos.tools.reason_hf import cosmos_reason_hf

# I/O + utilities
from strands_cosmos.tools.rtp import rtp_capture_frame
from strands_cosmos.tools.serve import cosmos_serve

# System
from strands_cosmos.tools.sysinfo import cosmos_sysinfo

# Transfer2.5 (ControlNet)
from strands_cosmos.tools.transfer_generate import cosmos_transfer_generate
from strands_cosmos.tools.video_utils import video_extract_frames, video_probe

__all__ = [
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
    # Legacy
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
