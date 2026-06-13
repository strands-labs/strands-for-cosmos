# 🧪 Strands Cosmos Examples

Copy-paste-ready scripts. Each one is the runnable distillation of a
[learning notebook](../notebooks) — **learn the concept in the notebook, ship it from here.**

> First time? Walk through [`../notebooks/00_start_here.ipynb`](../notebooks/00_start_here.ipynb)
> before diving in.

---

## Notebook → Example map

| Example script | Learn-first notebook | What it does |
|----------------|----------------------|--------------|
| `01_basic_text.py` | [01](../notebooks/01_basic_text.ipynb) | Text-only physics reasoning |
| `02_video_caption.py` | [02](../notebooks/02_video_caption.ipynb) | Caption a video with `<video>` |
| `03_driving_analysis.py` | [03](../notebooks/03_driving_analysis.ipynb) | Chain-of-thought driving safety |
| `04_embodied_reasoning.py` | [04](../notebooks/04_embodied_reasoning.ipynb) | Robot next-action from an image |
| `05_tool_usage.py` | [05](../notebooks/05_tool_usage.ipynb) | Cosmos as a callable tool |
| `06_cosmos3_reason.py` | [06](../notebooks/06_cosmos3_understand.ipynb) | Cosmos 3 reasoner (vLLM server) |
| `07_cosmos3_generate.py` | [07](../notebooks/07_cosmos3_generate.ipynb) | Cosmos 3 generator (image/video/sound) |
| `08_cosmos3_action.py` | [07](../notebooks/07_cosmos3_generate.ipynb) | World-model: forward dynamics |
| `09_cosmos3_showcase.py` | [06](../notebooks/06_cosmos3_understand.ipynb) + [07](../notebooks/07_cosmos3_generate.ipynb) | Full loop: reason → generate |
| `10_cosmos3_finetune.py` | *(advanced)* | SFT / fine-tuning via Cosmos Framework |

---

## Run an example

We use [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install strands-cosmos
python examples/01_basic_text.py
```

Most examples read the bundled `sample.mp4` / `sample.png` from the project root. Override with
environment variables:

```bash
SAMPLE_VIDEO=/path/to/clip.mp4 python examples/02_video_caption.py
SAMPLE_IMAGE=/path/to/photo.png python examples/04_embodied_reasoning.py
```

### Cosmos 3 examples (06–10) need their backends built once

```bash
just c3-doctor                                # check GPU / CUDA / disk
just c3-setup-reason && just c3-serve-reason  # 06, 09 (reasoner server on :8000)
just c3-setup-gen                             # 07, 09 (in-process Diffusers generator)
just c3-setup-framework                       # 08, 10 (world-model / training)
```

> 💡 On a single ~46 GB GPU, the reasoner and generator can't both be loaded at once — stop one
> before starting the other. (A Jetson AGX Thor with 128 GB can hold both.)
