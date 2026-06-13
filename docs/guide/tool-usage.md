# Tool Usage

Use Cosmos tools inside **any** Strands agent — Bedrock, Anthropic, OpenAI, Ollama, or any other provider.

!!! tip "Cosmos 3 tools"
    16 `cosmos3_*` tools are available the same way — reasoning (`cosmos3_caption`,
    `cosmos3_temporal`, ...), generation (`cosmos3_text2video`, `cosmos3_text2video_sound`),
    and action (`cosmos3_forward_dynamics`). See the [Cosmos 3 Guide](cosmos3.md).

---

## Overview

strands-cosmos provides **37 tools** (21 Cosmos-Reason2/Predict/Transfer + 16 Cosmos 3) that let any agent orchestrate the full NVIDIA Cosmos pipeline:

| Category | Tools |
|----------|-------|
| **Reason2 VLM** | `cosmos_inference`, `cosmos_reason_hf`, `cosmos_serve` |
| **World Models** | `cosmos_predict_generate`, `cosmos_transfer_generate` |
| **Model Lifecycle** | `cosmos_model_download`, `cosmos_quantize`, `cosmos_export_onnx`, `cosmos_build_engine` |
| **Training** | `cosmos_post_train`, `cosmos_distill` |
| **Data & Eval** | `cosmos_curate`, `cosmos_evaluate` |
| **I/O** | `rtp_capture_frame`, `nats_publish`, `video_probe`, `video_extract_frames`, `image_read` |
| **System** | `cosmos_sysinfo` |
| **Legacy** | `cosmos_invoke`, `cosmos_vision_invoke` |

---

## Basic Example

```python
from strands import Agent
from strands_cosmos import cosmos_reason_hf, video_probe, cosmos_sysinfo

# Create an agent with Cosmos tools (agent uses any model — Bedrock, OpenAI, etc.)
agent = Agent(tools=[cosmos_reason_hf, video_probe, cosmos_sysinfo])

# The agent decides when to call each tool
agent("Check the GPU, then analyze /tmp/dashcam.mp4 for safety hazards")
```

---

## Vision Tool (Direct VLM Inference)

```python
from strands import Agent
from strands_cosmos import cosmos_reason_hf

agent = Agent(tools=[cosmos_reason_hf])

# Video analysis
agent("Use cosmos_reason_hf to describe what's happening in /tmp/scene.mp4")

# Image analysis
agent("Use cosmos_reason_hf to identify objects in /tmp/workspace.jpg")
```

---

## Pipeline Orchestration

An agent can chain multiple tools to automate the full Cosmos workflow:

```python
from strands import Agent
from strands_cosmos import (
    cosmos_model_download,
    cosmos_quantize,
    cosmos_export_onnx,
    cosmos_build_engine,
    cosmos_serve,
    cosmos_inference,
)

agent = Agent(tools=[
    cosmos_model_download,
    cosmos_quantize,
    cosmos_export_onnx,
    cosmos_build_engine,
    cosmos_serve,
    cosmos_inference,
])

# Agent orchestrates the full edge-deployment pipeline
agent("""
Deploy Cosmos-Reason2-2B to this Jetson Thor:
1. Download the model
2. Quantize to FP8
3. Export to ONNX
4. Build TRT engine
5. Start the server
6. Run a test inference with "describe the scene"
""")
```

---

## Media Utilities

```python
from strands import Agent
from strands_cosmos import video_probe, video_extract_frames, image_read

agent = Agent(tools=[video_probe, video_extract_frames, image_read])

# Agent can analyze video metadata, extract frames, read images
agent("Probe /tmp/video.mp4, extract 4 frames at 1fps to /tmp/frames/, then read the first frame")
```

---

## System Diagnostics

```python
from strands import Agent
from strands_cosmos import cosmos_sysinfo

agent = Agent(tools=[cosmos_sysinfo])
agent("What GPU do I have and how much memory is available?")
```

---

## World Model Generation

```python
from strands import Agent
from strands_cosmos import cosmos_predict_generate

agent = Agent(tools=[cosmos_predict_generate])
agent("Generate future video frames using the config at /tmp/predict_config.json")
```

---

## All Tools in One Agent

```python
from strands import Agent
from strands_cosmos import *  # Import everything

# Create an omniscient Cosmos agent
agent = Agent(
    model=CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B"),
    tools=[
        cosmos_sysinfo, video_probe, video_extract_frames, image_read,
        cosmos_model_download, cosmos_predict_generate, cosmos_transfer_generate,
        cosmos_quantize, cosmos_export_onnx, cosmos_build_engine,
        cosmos_serve, cosmos_post_train, cosmos_distill,
        cosmos_curate, cosmos_evaluate, rtp_capture_frame, nats_publish,
    ],
)

agent("What can you do?")  # Agent will describe all available tools
```

---

## How Tools Work Internally

Each tool is a thin Python wrapper that delegates to `just <recipe>` commands:

```python
@tool
def cosmos_predict_generate(config_path: str) -> dict:
    """Generate video with Cosmos Predict2.5 world model."""
    return _run_just("predict-generate", config_path)
```

This means:
- Tools are **reproducible** — run the same `just` command manually
- Tools are **platform-aware** — justfile handles OS/GPU detection
- Tools **auto-clone** dependencies — `just ensure-predict` runs if repo is missing
