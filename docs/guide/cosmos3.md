# Cosmos 3 — Omnimodal World Models

[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) is NVIDIA's
omnimodal world-model family built on a unified **Mixture-of-Transformers (MoT)**
architecture that jointly processes and generates **language, images, video,
audio, and action sequences**. strands-cosmos provides first-class support for it
as Strands model providers + justfile-backed tools — **running entirely on local
compute.**

## Two runtime surfaces

| Surface | Inputs | Outputs | strands-cosmos artifact |
|---------|--------|---------|-------------------------|
| **Reasoner** | text, vision | text | `Cosmos3ReasonerModel` (vLLM) |
| **Generator** | text, vision, sound, action | vision, sound, action | `Cosmos3GeneratorModel` (Diffusers) + Cosmos Framework (action) |

## Model family

| Model | Size | Role |
|-------|-----:|------|
| [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) | 16B | Omnimodal — fits a single ~46GB GPU |
| [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) | 64B | Frontier-scale (multi-GPU / tensor-parallel) |
| [`nvidia/Cosmos3-Nano-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | 16B | VL robot policy (DROID) |

## Hardware & CUDA pairing

Cosmos 3 backends pin a CUDA build of `torch`/`vllm` that **must match your driver**:

| Driver CUDA | torch backend | vLLM |
|-------------|---------------|------|
| 13.x | `cu130` | `vllm==0.21.0` |
| 12.8 | `cu128` | `vllm==0.19.1` |

`just c3-doctor` reports your GPU, driver CUDA, the recommended pairing, venv
status, and free disk.

!!! warning "Single-GPU memory"
    The Reasoner (vLLM) and the Generator (Diffusers) each load a 16B model. On a
    single ~46GB GPU they **cannot run simultaneously** — stop one before starting
    the other, or dedicate separate GPUs.

---

## Reasoner — video & image understanding (vLLM)

```bash
just c3-setup-reason      # one-time: vllm==0.21.0 + vllm-cosmos3 (cu130)
just c3-serve-reason      # serve Cosmos3-Nano on :8000 (--max-model-len 32768)
```

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

agent = Agent(model=Cosmos3ReasonerModel(base_url="http://localhost:8000/v1"))

# Detailed captioning
agent("Caption in detail: <video>scene.mp4</video>")

# Temporal localization with timestamps
agent("List the notable events with approximate timestamps: <video>scene.mp4</video>")

# Embodied next-action with explicit reasoning
agent.model.update_config(reasoning=True)
agent("What is the most likely next action? <video>robot.mp4</video>")
```

Reasoner capabilities (each has a dedicated tool):

| Tool | Task |
|------|------|
| `cosmos3_caption` | Detailed captioning |
| `cosmos3_temporal` | Event detection + timestamps |
| `cosmos3_embodied` | Next-action prediction |
| `cosmos3_ground` | 2D bounding boxes (JSON) |
| `cosmos3_plausibility` | Physical plausibility label |
| `cosmos3_situation` | Situation understanding |
| `cosmos3_action_cot` | Trajectory / driving CoT |

---

## Generator — image, video & sound (Diffusers, in-process)

```bash
# Option A — pip extra (Diffusers + cosmos_guardrail + soundfile):
pip install "strands-cosmos[cosmos3-gen]"
# Cosmos3OmniPipeline needs the diffusers dev build:
pip install -U "git+https://github.com/huggingface/diffusers.git"

# Option B — justfile (dedicated CUDA-matched venv):
just c3-setup-gen   # diffusers(main) + cosmos_guardrail + soundfile (cu130)
```

```python
from strands_cosmos import Cosmos3GeneratorModel
m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")

# Text → image
m.generate(mode="text2image", prompt="A robot in a warehouse.",
           out_path="img.png", resolution="480")

# Text → video
m.generate(mode="text2video", prompt="A robot navigates a warehouse aisle.",
           out_path="vid.mp4", num_frames=49, fps=16, num_inference_steps=25)

# Image → video
m.generate(mode="image2video", prompt="It begins to move forward.",
           image="img.png", out_path="i2v.mp4")

# Text → video WITH SOUND (H264 video + AAC stereo 48kHz)
m.generate(mode="text2video-with-sound", prompt="A robot arm pours water.",
           out_path="av.mp4", enable_sound=True)
```

Sound is generated in-process by the omni pipeline (`Cosmos3OmniPipelineOutput`
returns both `video` frames and a stereo `sound` tensor); strands-cosmos muxes it
into the MP4 via ffmpeg. No vLLM-Omni server is required for image/video/sound.

---

## Action / World-Model (Cosmos Framework)

Action generation (forward/inverse dynamics, policy) runs through the native
Cosmos Framework via `torchrun`.

```bash
just c3-setup-framework   # one-time: clone cosmos-framework + uv sync (cu130-train)
```

Each run is described by a **JSONL spec** (one line per run):

```json
{
  "model_mode": "forward_dynamics",
  "name": "av_forward",
  "vision_path": ".../images/av_0.jpg",
  "action_path": ".../actions/av_traj_forward.json",
  "domain_name": "av",
  "action_chunk_size": 60,
  "fps": 10,
  "image_size": 480,
  "view_point": "ego_view",
  "prompt": "You are an autonomous vehicle planning system.",
  "seed": 0
}
```

```python
from strands_cosmos import cosmos3_forward_dynamics
cosmos3_forward_dynamics(input_jsonl="fd_av.jsonl", out="/tmp/c3_action_out")
# → /tmp/c3_action_out/av_forward/vision.mp4  (future-state rollout)
```

| Tool | Task |
|------|------|
| `cosmos3_forward_dynamics` | start image + action chunk → future video |
| `cosmos3_inverse_dynamics` | video + instruction → predicted action chunk |
| `cosmos3_policy` | image + instruction → action chunk + rollout video |

Embodiments include autonomous vehicle (`av`), DROID/Bridge/UMI single-arm robots,
dual-arm, and humanoid — see the
[Cosmos cookbooks](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action).

---

## justfile reference

```bash
just c3-doctor             # environment check (GPU/CUDA/uv/venvs/disk)
just c3-setup-reason       # Reasoner env (vLLM + vllm-cosmos3)
just c3-setup-gen          # Generator env (Diffusers)
just c3-setup-framework    # Action env (Cosmos Framework)
just c3-serve-reason       # start Cosmos3-Nano reasoner server
just c3-serve-stop-reason  # stop it
just c3-serve-status       # server status
just c3-reason "<prompt>" "<image>" "<video>" "<task>"
just c3-gen <mode> "<prompt>" "<image>" <out>
just c3-action <input.jsonl> <out>
```
