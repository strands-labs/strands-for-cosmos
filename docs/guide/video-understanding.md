# Video Understanding

Cosmos-Reason2 processes video at configurable frame rates, extracting temporal and spatial information.

!!! tip "Also available with Cosmos 3"
    The latest **Cosmos 3** Reasoner does the same video understanding (caption, temporal
    localization, embodied next-action, grounding) via `Cosmos3ReasonerModel` — and can
    then **generate** new video/audio. See the [Cosmos 3 Guide](cosmos3.md).

---

![Video Understanding Pipeline](../assets/svg/video-flow.svg)


## See It In Action

<img src="/strands-cosmos/assets/videos/02_video_caption.gif" alt="Video captioning on Jetson AGX Thor" width="100%">

<details>
<summary>📺 Can't see the animation? <a href="/strands-cosmos/assets/videos/02_video_caption.mp4">Download MP4</a></summary>

<video controls width="100%" muted>
  <source src="/strands-cosmos/assets/videos/02_video_caption.mp4" type="video/mp4">
</video>

</details>

---

## How Video Processing Works

```mermaid
graph TD
    V["🎬 Video File"] --> D["Decode Frames<br/>torchvision / av"]
    D --> S["Sample @ FPS<br/>default: 4 fps"]
    S --> T["Visual Tokens<br/>256–8192 per frame"]
    T --> M["Cosmos-Reason2<br/>Qwen3-VL backbone"]
    M --> R["Response"]
```

## Basic Video Captioning

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    fps=4,
    params={"max_tokens": 4096},
)
agent = Agent(model=model)

agent("Caption this video in detail: <video>scene.mp4</video>")
```

→ [Full video captioning example](../examples/video-caption.md)

## Driving Analysis (Video + CoT)

<img src="/strands-cosmos/assets/videos/03_driving_analysis.gif" alt="Driving analysis with chain-of-thought" width="100%">

```python
model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    fps=4,
    params={"max_tokens": 4096, "temperature": 0.6},
)
agent = Agent(model=model)

agent("""<video>highway.mp4</video>
You are an expert driving assistant. Analyze:
1. Current road conditions
2. Potential hazards
3. Recommended actions for the driver""")
```

→ [Full driving analysis example](../examples/driving.md)

## Configuring Frame Rate

Higher FPS = more detail but more GPU memory and slower inference.

| FPS | Frames (10s video) | Use Case |
|-----|-------------------|----------|
| 1 | 10 | Quick summaries |
| 4 | 40 | **Default — balanced** |
| 8 | 80 | Detailed temporal analysis |

```python
model = CosmosVisionModel(fps=8)  # Higher detail
```

## Controlling Visual Token Budget

```python
model = CosmosVisionModel(
    min_vision_tokens=256,   # Minimum per frame
    max_vision_tokens=8192,  # Maximum per frame
)
```

## Built-in Task Prompts

Cosmos includes optimized prompts for common tasks:

| Task Key | Description |
|----------|-------------|
| `caption` | Detailed video/image captioning |
| `driving` | Dashcam driving analysis |
| `embodied_reasoning` | Robot next-action prediction |
| `causal` | Physical cause-and-effect reasoning |
| `temporal_localization` | Event timestamps in video |
| `2d_grounding` | Bounding box localization |
| `robot_cot` | Step-by-step robot planning |

```python
from strands_cosmos.cosmos_vision_model import TASK_PROMPTS

# Use a task prompt directly
prompt = TASK_PROMPTS["driving"]
agent(f"{prompt} <video>dashcam.mp4</video>")
```

---

## What's Next

- [**Image Reasoning**](image-reasoning.md) — Single-frame analysis
- [**Chain-of-Thought**](chain-of-thought.md) — Enable step-by-step reasoning
- [**Examples**](../examples/overview.md) — Runnable code for all scenarios
