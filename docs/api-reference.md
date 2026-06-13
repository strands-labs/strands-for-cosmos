# API Reference

## Models

### `CosmosVisionModel`

The primary model class — supports video, image, and text input.

```python
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id: str = "nvidia/Cosmos-Reason2-2B",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    reasoning: bool = False,
    fps: int = 4,
    min_vision_tokens: int = 256,
    max_vision_tokens: int = 8192,
    params: dict = {},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `nvidia/Cosmos-Reason2-2B` | HuggingFace model ID |
| `device_map` | `str` | `auto` | GPU device placement |
| `torch_dtype` | `str` | `auto` | Tensor dtype (float16/bfloat16) |
| `reasoning` | `bool` | `False` | Enable chain-of-thought `<think>` reasoning |
| `fps` | `int` | `4` | Video frame sampling rate |
| `min_vision_tokens` | `int` | `256` | Minimum visual tokens per frame |
| `max_vision_tokens` | `int` | `8192` | Maximum visual tokens per frame |
| `params` | `dict` | `{}` | Generation params: `max_tokens`, `temperature`, `top_p` |

### `CosmosModel`

Text-only model — same interface but no vision capabilities.

```python
from strands_cosmos import CosmosModel

model = CosmosModel(model_id="nvidia/Cosmos-Reason2-2B")
```

### `Cosmos3ReasonerModel`

**NEW (Cosmos 3).** Omnimodal Reasoner — text + vision → text — served by a local
vLLM server. Captioning, temporal localization, embodied next-action,
2D grounding, physical plausibility, situation understanding, action CoT.

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

model = Cosmos3ReasonerModel(
    model_id: str = "nvidia/Cosmos3-Nano",
    base_url: str = "http://localhost:8000/v1",
    reasoning: bool = False,        # explicit <think> reasoning
    max_tokens: int = 4096,
    seed: int | None = 0,
    media_io_kwargs: dict | None = None,    # e.g. {"video": {"fps": 4.0}}
    mm_processor_kwargs: dict | None = None, # e.g. {"size": {"shortest_edge": 1568}}
)
agent = Agent(model=model)
agent("Caption in detail: <video>scene.mp4</video>")
```

Start the server first:

```bash
just c3-setup-reason      # one-time: vllm==0.21.0 + vllm-cosmos3 (cu130)
just c3-serve-reason      # serve Cosmos3-Nano on :8000 (--max-model-len 32768)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `nvidia/Cosmos3-Nano` | Served model (auto-resolved from server) |
| `base_url` | `str` | `http://localhost:8000/v1` | vLLM OpenAI endpoint |
| `reasoning` | `bool` | `False` | Append `<think>` format + use reasoning sampling preset |
| `max_tokens` | `int` | `4096` | Output token cap |
| `media_io_kwargs` | `dict` | `None` | Video frame sampling passthrough |
| `mm_processor_kwargs` | `dict` | `None` | Per-image resize bounds passthrough |

Inline media tags: `<video>path-or-url</video>`, `<image>path-or-url</image>`.

### `Cosmos3GeneratorModel`

**NEW (Cosmos 3).** Omnimodal Generator — text/image → image/video/**sound** —
in-process via HuggingFace Diffusers `Cosmos3OmniPipeline` (no server).

```python
from strands_cosmos import Cosmos3GeneratorModel

m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")

m.generate(mode="text2image",  prompt="A robot in a warehouse.",
           out_path="img.png", resolution="480")
m.generate(mode="text2video",  prompt="A robot navigates a warehouse.",
           out_path="vid.mp4", num_frames=49, fps=16, num_inference_steps=25)
m.generate(mode="image2video", prompt="It starts moving.", image="img.png",
           out_path="i2v.mp4")
m.generate(mode="text2video-with-sound", prompt="A robot pours water.",
           out_path="av.mp4", enable_sound=True)   # H264 + AAC stereo 48kHz
```

Setup: `just c3-setup-gen` (diffusers main + cosmos_guardrail, cu130).

| `generate()` arg | Type | Default | Description |
|------------------|------|---------|-------------|
| `mode` | `str` | `text2video` | `text2image` / `text2video` / `image2video` / `text2video-with-sound` |
| `prompt` | `str` | `""` | Positive text prompt |
| `out_path` | `str` | `/tmp/cosmos3_out.mp4` | Output file (`.png` for image) |
| `image` | `str` | `None` | Input image (image2video) |
| `num_frames` | `int` | `189` | Frame count (1 for image) |
| `fps` | `int` | `24` | Frames per second |
| `resolution` | `str` | `720` | `256` / `480` / `720` |
| `num_inference_steps` | `int` | `35` | Diffusion steps |
| `guidance_scale` | `float` | `6.0` | CFG scale |
| `enable_sound` | `bool` | `False` | Generate + mux stereo audio (AAC 48kHz) |
| `seed` | `int` | `0` | Reproducibility seed |

> **Single-GPU note:** the reasoner (vLLM) and generator (Diffusers) each load a
> 16B model — they won't fit on one ~46GB GPU together. Stop one before the other.

---

## Tools

All tools are `@tool`-decorated functions compatible with any Strands Agent.

### Reason2 VLM

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_inference` | `prompt`, `image_path?`, `video_path?`, `server_url?` | Query TRT-Edge-LLM inference server |
| `cosmos_reason_hf` | `prompt`, `image_path?`, `video_path?`, `max_new_tokens?`, `model_id?` | Direct HF Transformers inference (no server needed) |
| `cosmos_serve` | `action` (`start`/`stop`/`status`) | Manage TRT-Edge-LLM server lifecycle |

### World Models

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_predict_generate` | `config_path` | Generate future video frames with Predict2.5 |
| `cosmos_transfer_generate` | `config_path` | Video-to-video with Transfer2.5 (ControlNet) |

### Model Lifecycle

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_model_download` | `name`, `local_dir?`, `kind?` | Download model from HuggingFace |
| `cosmos_quantize` | `model_dir`, `output_dir?`, `precision?` | FP8/INT8 quantization |
| `cosmos_export_onnx` | `model_dir`, `output_dir?` | Export to ONNX format |
| `cosmos_build_engine` | `onnx_dir`, `output_dir?`, `component?` | Build TRT engine (LLM or visual) |

### Training

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_post_train` | `config_path`, `method?` | Post-training (SFT, LoRA, full) |
| `cosmos_distill` | `config_path` | Knowledge distillation (8B→2B) |

### Data & Evaluation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_curate` | `config_path` | Run Xenna data curation pipeline |
| `cosmos_evaluate` | `config_path`, `metrics?` | Evaluate with FID/FVD/CSE/CLIP |

### I/O & Media

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rtp_capture_frame` | `port?`, `output_path?` | Capture single frame from RTP/GStreamer stream |
| `nats_publish` | `subject`, `payload` | Publish JSON to NATS subject |
| `video_probe` | `video_path` | Get video metadata (resolution, fps, duration, codec) |
| `video_extract_frames` | `video_path`, `output_dir`, `fps?`, `max_frames?` | Extract frames as JPEGs |
| `image_read` | `image_path` | Read image as base64 string |

### System

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_sysinfo` | — | GPU info, platform, memory, CUDA version |


### Cosmos 3 — Reasoner (vLLM)

| Tool | Description |
|------|-------------|
| `cosmos3_reason` | Generic reasoner: prompt + image/video → text |
| `cosmos3_caption` | Detailed video/image captioning |
| `cosmos3_temporal` | Event detection + timestamps |
| `cosmos3_embodied` | Next-action prediction (robotics) |
| `cosmos3_ground` | 2D bounding-box grounding (JSON) |
| `cosmos3_plausibility` | Physical plausibility classification |
| `cosmos3_situation` | Situation understanding + next action |
| `cosmos3_action_cot` | Trajectory / driving chain-of-thought |

### Cosmos 3 — Generator (Diffusers, in-proc)

| Tool | Description |
|------|-------------|
| `cosmos3_text2image` | Text → image (PNG) |
| `cosmos3_text2video` | Text → video (MP4) |
| `cosmos3_image2video` | Image + text → video |
| `cosmos3_text2video_sound` | Text → video + synchronized audio (AAC stereo 48kHz) |
| `cosmos3_video2video` | Re-render an input video with a new prompt (transfer; vLLM-Omni Docker) |

### Cosmos 3 — Action / World-Model (Cosmos Framework)

| Tool | Description |
|------|-------------|
| `cosmos3_forward_dynamics` | Start image + action chunk → future video |
| `cosmos3_inverse_dynamics` | Video + instruction → predicted action chunk |
| `cosmos3_policy` | Image + instruction → action chunk + rollout video |

### Cosmos 3 — Servers

| Tool | Description |
|------|-------------|
| `cosmos3_serve` | Start/stop/status local vLLM (reason) / vLLM-Omni (omni) servers |

### Cosmos 3 — Post-Training (SFT)

Supervised fine-tuning via the Cosmos Framework (`torchrun`). Tested upstream on 8× H100.

| Tool | Description |
|------|-------------|
| `cosmos3_train_recipes` | List SFT recipes + launch shells |
| `cosmos3_train_show` | Validate/print a recipe's resolved config (dry run) |
| `cosmos3_train_convert` | Base checkpoint → PyTorch DCP |
| `cosmos3_train_convert_vlm` | LM → Qwen3-VL visual tower (reasoner VLM) |
| `cosmos3_train_prep_dataset` | captions JSONL → SFT dataset JSONL |
| `cosmos3_train` | Run SFT via the paired launch shell |
| `cosmos3_train_export` | Trained DCP → HF safetensors |

See the [Cosmos 3 Training guide](guide/cosmos3-training.md) for the full flow.

### Legacy (backward-compatible)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_invoke` | `prompt`, `model_id?` | Text-only inference tool |
| `cosmos_vision_invoke` | `prompt`, `media_path?`, `model_id?` | Vision inference tool |

---

## Task Prompts

Pre-defined prompts optimized for specific tasks:

```python
from strands_cosmos.cosmos_vision_model import TASK_PROMPTS
```

| Key | Use Case |
|-----|----------|
| `caption` | Detailed video/image captioning |
| `embodied_reasoning` | Robot workspace analysis |
| `driving` | Dashcam driving safety |
| `causal` | Physical cause-and-effect |
| `temporal_localization` | Event timestamps in video |
| `2d_grounding` | Bounding box coordinates |
| `robot_cot` | Step-by-step robot planning |
| `describe_anything` | General scene description |
| `mvp_bench` | MVP benchmark evaluation |

---

## CLI

### `strands-cosmos-fix-cublas`

Fix CUBLAS compatibility on NVIDIA Jetson devices.

```bash
strands-cosmos-fix-cublas           # Auto-detect and fix
strands-cosmos-fix-cublas --check   # Check status only
strands-cosmos-fix-cublas --revert  # Restore original
```

---

## Justfile Recipes

Run `just --list` for all available recipes. Key ones:

```bash
just setup          # Clone all Cosmos ecosystem repos
just setup-full     # Full setup (apt + pip + repos + doctor)
just doctor         # Diagnose platform, tools, GPU
just install-trt-edge-llm  # Build TRT-Edge-LLM from source

just serve-start    # Start TRT inference server
just serve-stop     # Stop server
just predict-generate config.json
just transfer-generate config.json
just evaluate config.json
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COSMOS_MODEL_ID` | Default HF model | `nvidia/Cosmos-Reason2-2B` |
| `COSMOS_SERVER_URL` | TRT server endpoint | `http://127.0.0.1:8080` |
| `NATS_URL` | NATS server URL | `nats://127.0.0.1:4222` |
| `RTP_PORT` | RTP receive port | `5600` |
| `HF_TOKEN` | HuggingFace token for gated models | — |
| `COSMOS_PREDICT_REPO` | Path to cosmos-predict2.5 clone | `../cosmos-predict2.5` |
| `COSMOS_TRANSFER_REPO` | Path to cosmos-transfer2.5 clone | `../cosmos-transfer2.5` |
| `COSMOS_REASON_REPO` | Path to cosmos-reason2 clone | `../cosmos-reason2` |
| `COSMOS_XENNA_REPO` | Path to cosmos-xenna clone | `../cosmos-xenna` |
| `COSMOS_COOKBOOK_REPO` | Path to cosmos-cookbook clone | `../cosmos-cookbook` |
