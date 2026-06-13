# Image Reasoning

Cosmos-Reason2 processes single images for object recognition, spatial reasoning, and embodied intelligence.

!!! tip "Also available with Cosmos 3"
    **Cosmos 3** supports image reasoning (grounding, describe-anything, action CoT) via
    `Cosmos3ReasonerModel`, plus image **generation** and image→video. See the
    [Cosmos 3 Guide](cosmos3.md).

---

## See It In Action

<img src="/strands-cosmos/assets/videos/04_embodied_reasoning.gif" alt="Embodied robot reasoning from image" width="100%">

<details>
<summary>📺 Can't see the animation? <a href="/strands-cosmos/assets/videos/04_embodied_reasoning.mp4">Download MP4</a></summary>

<video controls width="100%" muted>
  <source src="/strands-cosmos/assets/videos/04_embodied_reasoning.mp4" type="video/mp4">
</video>

</details>

---

## Basic Image Analysis

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

agent("<image>workspace.jpg</image> Describe what you see.")
```

## Embodied Reasoning (Robot Vision)

```python
model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
)
agent = Agent(model=model)

agent("""<image>robot_view.jpg</image>
Given this view from a bimanual robot's workspace:
What is the immediate next action the robot should take?""")
```

The model will reason through `<think>...</think>` tags before providing the action.

→ [Full embodied reasoning example](../examples/embodied.md)

## 2D Grounding

Cosmos can localize objects with bounding box coordinates:

```python
agent("""<image>kitchen.jpg</image>
Locate the red cup in this image. Provide bounding box coordinates.""")
```

## Image Format Support

Images are processed via the Qwen3-VL processor:

| Format | Supported |
|--------|-----------|
| JPEG / JPG | ✅ |
| PNG | ✅ |
| WebP | ✅ |
| BMP | ✅ |

## Visual Token Configuration

```python
model = CosmosVisionModel(
    min_vision_tokens=256,    # Minimum visual detail
    max_vision_tokens=8192,   # Maximum visual detail
)
```

Higher `max_vision_tokens` = more detail at the cost of memory and speed.

---

## What's Next

- [**Video Understanding**](video-understanding.md) — Multi-frame temporal analysis
- [**Chain-of-Thought**](chain-of-thought.md) — Step-by-step reasoning
