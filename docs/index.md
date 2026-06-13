<div align="center">
  <img src="strands-cosmos-logo.svg" alt="Strands Cosmos" width="180">
  <h1>Strands Cosmos</h1>
  <p><strong>NVIDIA Cosmos for Strands Agents — omnimodal world-model reasoning <em>and</em> generation, on local compute.</strong></p>
</div>

Cosmos models become first-class **Strands model providers**: give your agent eyes that
understand physics, and hands that can generate video, audio, and robot actions — plus
**45 tools** spanning the full Cosmos pipeline.

| Family | Providers | Best for |
|--------|-----------|----------|
| **Cosmos 3** (latest, omnimodal) | `Cosmos3ReasonerModel`, `Cosmos3GeneratorModel` | Video/image/audio/action **understanding + generation** |
| **Cosmos-Reason2** (VLM) | `CosmosVisionModel`, `CosmosModel` | Lightweight edge VLM (Jetson Thor/Orin) |

---

## 🌌 Cosmos 3 — Reason → Generate

[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) is NVIDIA's newest model
family — a unified **Mixture-of-Transformers** that jointly **understands and generates**
text, images, video, audio, and action. Here it watches a real construction-site clip,
**describes it**, then **generates new videos** (one with synchronized audio) from its own
description — all on a single local GPU.

<table>
<tr><th>① Input video</th><th>② Cosmos 3 understands it</th></tr>
<tr>
<td><img src="assets/cosmos3_showcase/00_input.gif" width="200" alt="input"/></td>
<td>

> *"Two construction workers wearing yellow safety vests and helmets are walking away from the camera on a dirt path within a bustling construction site. The ground is covered in loose soil, with visible tire tracks crisscrossing the surface. In the background, a large yellow front-end loader moves slowly across the site, its bucket raised slightly as it navigates the terrain. Behind the loader, partially obscured by rebar and concrete slabs, an excavator operates near a foundation area. The scene is framed by urban buildings in the distance, including a distinctive church-like structure with a tall spire and modern glass-fronted buildings. The overall atmosphere suggests active progress on a significant infrastructure project under clear daylight conditions."*

— `Cosmos3ReasonerModel` (caption in 5.2s)

</td>
</tr>
</table>

The reasoner distills its own understanding into a generation prompt:

> **"Two construction workers in yellow safety vests and helmets walk across a dusty site, gesturing toward a yellow front loader and distant excavator as they converse."**

Then `Cosmos3GeneratorModel` generates similar videos from that prompt (832×480, 49 frames):

<table>
<tr><th>text → video</th><th>text → video + 🔊 sound</th><th>image → video</th></tr>
<tr>
<td><img src="assets/cosmos3_showcase/01_text2video.gif" width="220" alt="text2video"/></td>
<td><img src="assets/cosmos3_showcase/02_text2video_sound.gif" width="220" alt="text2video+sound"/></td>
<td><img src="assets/cosmos3_showcase/03_image2video.gif" width="220" alt="image2video"/></td>
</tr>
<tr>
<td align="center"><sub>55.5s</sub></td>
<td align="center"><sub>43.2s · AAC stereo 48kHz</sub></td>
<td align="center"><sub>42.1s · from a real frame</sub></td>
</tr>
</table>

→ Full walkthrough: **[Cosmos 3 Guide](guide/cosmos3.md)** · reproduce with `python examples/09_cosmos3_showcase.py`

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel, Cosmos3GeneratorModel

# Reasoner — text + vision -> text (local vLLM; start with `just c3-serve-reason`)
agent = Agent(model=Cosmos3ReasonerModel(base_url="http://localhost:8000/v1"))
agent("Caption in detail: <video>scene.mp4</video>")

# Generator — text/image -> image/video/sound (in-process Diffusers, no server)
gen = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")
gen.generate(mode="text2video-with-sound", prompt="A robot pours water.",
             out_path="av.mp4", enable_sound=True)   # H264 + AAC stereo 48kHz
```

!!! warning "Single-GPU note"
    The reasoner (vLLM) and generator (Diffusers) each load a 16B model — on one
    ~46GB GPU, stop one before running the other. CUDA pairing: CUDA 13 → `cu130` +
    `vllm==0.21.0`. `just c3-doctor` reports your driver's recommendation.

---

## Cosmos-Reason2 — Lightweight Edge VLM

For edge/Jetson, the Cosmos-Reason2 VLM runs as a model provider with a tiny footprint —
verified on Jetson AGX Thor with Chain-of-Thought reasoning.

<div class="grid cards" markdown>

- **🚗 Driving Analysis with Chain-of-Thought**

    <img src="/strands-cosmos/assets/videos/03_driving_analysis.gif" alt="Driving analysis with CoT reasoning" width="100%">

    → [Full example + code](examples/driving.md)

- **🤖 Robot Embodied Reasoning**

    <img src="/strands-cosmos/assets/videos/04_embodied_reasoning.gif" alt="Robot embodied reasoning" width="100%">

    → [Full example + code](examples/embodied.md)

</div>

<div class="grid cards" markdown>

- **🎬 Video Captioning**

    <img src="/strands-cosmos/assets/videos/02_video_caption.gif" alt="Video captioning" width="100%">

    → [Full example + code](examples/video-caption.md)

- **⚛️ Physics Reasoning (Text-Only)**

    <img src="/strands-cosmos/assets/videos/01_basic_text.gif" alt="Physics reasoning" width="100%">

    → [Full example + code](examples/basic-text.md)

</div>

```mermaid
graph LR
    A["🗣️ Strands Agent"] --> RCosmos 3
    R -->|Reasoner| U["📹 Understand: caption · temporal · embodied · grounding"]
    R -->|Generator| G["🎬 Generate: image · video · 🔊 audio · 🤖 action"]
    A --> VCosmos-Reason2 VLM
    V -->|Edge| E["🚗 Driving · Robot planning · CoT"]
```

---

## Get Started in 2 Minutes

```bash
pip install strands-cosmos
```

For **Cosmos 3**, see the [Cosmos 3 Guide](guide/cosmos3.md) (`just c3-setup-reason` / `just c3-setup-gen`).
For the **Reason2 edge VLM**, it works straight from `pip`:

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

# Analyze a dashcam video
agent("Caption in detail: <video>dashcam.mp4</video>")

# Reason about a robot's view
agent("<image>robot_view.jpg</image> What should the robot do next?")

# Physics understanding (text-only)
agent("What happens when you push a ball off the edge of a table?")
```

→ **[Full Quickstart](getting-started/quickstart.md)** | **[Installation](getting-started/installation.md)**

---

## Capabilities

<div class="grid cards" markdown>

- **🚗 Driving Analysis**

    Traffic, hazards, navigation from dashcam video

    → [Driving example](examples/driving.md)

- **🤖 Robot Planning**

    Next-action prediction, 2D trajectory planning

    → [Embodied reasoning](examples/embodied.md)

- **🎬 Video Captioning**

    Detailed temporal-spatial descriptions

    → [Video captioning](examples/video-caption.md)

- **⚛️ Physics Reasoning**

    Object permanence, causality, plausibility

    → [Text reasoning](examples/basic-text.md)

- **🔍 2D Grounding**

    Bounding box localization in images

- **🧠 Chain-of-Thought**

    `<think>` reasoning before answers

    → [CoT guide](guide/chain-of-thought.md)

</div>

---

## Models

**Cosmos 3 (omnimodal — reasoning + generation):**

| Model | Size | Capability |
|-------|-----:|------------|
| [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | 16B | Omnimodal (reasoner + generator + action) — fits a single ~46GB GPU |
| [Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) | 64B | Frontier-scale (multi-GPU / tensor-parallel) |
| [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | 16B | VL robot policy (DROID) |

**Cosmos-Reason2 (lightweight edge VLM):**

| Model | GPU Memory | Architecture | Best For |
|-------|-----------|--------------|----------|
| [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | 24 GB | Qwen3-VL | Edge / Jetson |
| [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | 32 GB | Qwen3-VL | Desktop / Cloud |

### Verified Platforms

| Platform | GPU | Status |
|----------|-----|--------|
| Jetson AGX Thor | Thor 132 GB | ✅ (with CUBLAS fix) |
| Desktop | A100 / H100 / RTX 4090 | ✅ |
| Jetson Orin | Orin 32/64 GB | ✅ (may need CUBLAS fix) |

---

## Two Ways to Use

=== "As the Agent's Model"
    ```python
    from strands import Agent
    from strands_cosmos import CosmosVisionModel

    model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
    agent = Agent(model=model)
    agent("Describe this scene: <video>scene.mp4</video>")
    ```

=== "As a Tool (in any Agent)"
    ```python
    from strands import Agent
    from strands_cosmos import cosmos_reason_hf, video_probe, cosmos_sysinfo

    # 45 tools available — use any combination
    agent = Agent(tools=[cosmos_reason_hf, video_probe, cosmos_sysinfo])
    agent("Check GPU status, probe the video, then describe what you see in /tmp/scene.mp4")
    ```

=== "Full Pipeline (Agent automates Cosmos)"
    ```python
    from strands import Agent
    from strands_cosmos import (
        cosmos_model_download, cosmos_quantize, cosmos_export_onnx,
        cosmos_build_engine, cosmos_serve, cosmos_inference,
    )

    # Agent orchestrates the full edge-deployment pipeline
    agent = Agent(tools=[
        cosmos_model_download, cosmos_quantize, cosmos_export_onnx,
        cosmos_build_engine, cosmos_serve, cosmos_inference,
    ])
    agent("Download Reason2-2B, quantize to FP8, export ONNX, build TRT engine, start server, and run a test query")
    ```

---

## Performance on Jetson AGX Thor

Benchmarks with Cosmos-Reason2-2B on 132GB unified memory:

| Example | Task | Time | Recording |
|---------|------|------|-----------|
| 01 | Text-only physics | ~11s | [:material-play: cast](assets/casts/01_basic_text.cast) |
| 02 | Video caption (10s @ 4fps) | ~15s | [:material-play: cast](assets/casts/02_video_caption.cast) |
| 03 | Driving analysis + CoT | ~16s | [:material-play: cast](assets/casts/03_driving_analysis.cast) |
| 04 | Embodied reasoning + CoT | ~43s | [:material-play: cast](assets/casts/04_embodied_reasoning.cast) |
| 05 | Tool invocation | ~9s | [:material-play: cast](assets/casts/05_tool_usage.cast) |

---

## Quick Links

<div class="grid" markdown>

[:material-download: **Installation** →](getting-started/installation.md)

[:material-rocket-launch: **Quickstart** →](getting-started/quickstart.md)

[:material-video: **Video Understanding** →](guide/video-understanding.md)

[:material-brain: **Chain-of-Thought** →](guide/chain-of-thought.md)

[:material-tools: **Tool Usage** →](guide/tool-usage.md)

[:material-chip: **Jetson Deployment** →](guide/jetson.md)

[:material-file-tree: **Architecture** →](architecture.md)

[:material-code-tags: **API Reference (45 tools)** →](api-reference.md)

</div>

---

## Developer Setup (Full Cosmos Ecosystem)

```bash
git clone https://github.com/cagataycali/strands-cosmos && cd strands-cosmos
just setup-full    # Installs apt deps, Python deps, clones 6 Cosmos repos
just doctor        # Platform diagnostics — what works on THIS machine
```

`just doctor` checks: repos, core tools, Python packages, media tools, TRT binaries, GPU/CUDA — with platform-aware guidance (workstation vs Jetson vs Docker).

---

## Resources

- [Cosmos-Reason2 GitHub](https://github.com/nvidia-cosmos/cosmos-reason2)
- [HuggingFace Models](https://huggingface.co/collections/nvidia/cosmos-reason2)
- [Strands Agents](https://strandsagents.com)
- [PyPI Package](https://pypi.org/project/strands-cosmos/)
