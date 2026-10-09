# Cosmos3-Edge (4B) — reasoner, generator and action on one edge GPU

[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) is the 4B member of the Cosmos 3
family. It differs from Cosmos3-Nano in three ways that matter here:

| | Cosmos3-Nano (16B) | Cosmos3-Edge (4B) |
|---|---|---|
| Reasoner path in this package | vLLM server (`Cosmos3ReasonerModel`) | **in-process Transformers** (`Cosmos3EdgeHFModel`) *or* vLLM |
| Generator | Diffusers `Cosmos3OmniPipeline` | same class, released `diffusers>=0.40` |
| Action | Cosmos Framework (`just c3-action`) | **in-process Diffusers** (`cosmos3_action_edge`) *or* Framework |
| Sound | yes | **no** (`sound_gen=False`) — the `*_sound` tools refuse it before loading |
| Weights on GPU (bf16) | ~46 GB | 4.55 GiB understanding / 7.61 GiB generation pipeline |
| Licence | NVIDIA Open Model License | OpenMDW-1.1 (permissive; the repo ships no LICENSE file, only the README line) |

Everything below was **measured** on a Jetson AGX Thor (arm64, CUDA sm_110, 128 GB unified memory),
`torch 2.14.1+cu130 · transformers 5.19.0 · diffusers 0.41.0`, Edge revision `344d602b…`, `HF_HUB_OFFLINE=1`.
Memory is `torch.cuda.max_memory_allocated()` (nvidia-smi reports N/A on Jetson).

## Setup

```bash
just c3-setup-edge        # .venv-c3-edge: torch(cu130) + transformers>=5.19 + diffusers>=0.40 + torchcodec/av
just c3-edge-doctor       # prints what loads, which embodiments are trained
```

`transformers 5.2` / `diffusers 0.35` cannot load this checkpoint (no `cosmos3_edge` model, no
`Cosmos3OmniTransformer`); the provider raises a clear `ImportError` instead of a key error.
Video input needs `torchcodec` (or `av`) — without them `AutoProcessor` fails with an ImportError.

## Reasoner — `Cosmos3EdgeHFModel`

```python
from strands import Agent
from strands_cosmos import Cosmos3EdgeHFModel

agent = Agent(model=Cosmos3EdgeHFModel())                    # lazy load on first call
agent("Caption the image in detail: <image>ws_red.png</image>")
agent("Caption the video in detail: <video>sample.mp4</video>")
agent("In one sentence: what kind of camera recorded that video?")   # -> "A dashcam"
```

Edge answers inside a `<think>…</think>` block first; the provider streams that as Strands
`reasoningContent` and only the answer as text (`show_thinking=True` keeps the raw stream).
Media paths are resolved against `COSMOS_WORKSPACE` (default: the current directory) — a path outside
it is dropped with a warning and the model will answer *without* the image, so set the workspace.

Also available from the tool: `cosmos3_reason(prompt, image=..., backend="hf")`.

| Thor measurement (`Agent`, max_tokens 512) | wall | tokens in / out | peak GPU |
|---|---|---|---|
| load (`from_pretrained` → cuda) | 9.4–9.8 s | — | 4.549 GiB weights |
| image caption, cold / warm (raw generate) | 5.0 s / 3.9–4.5 s · 15.3→20.0 tok/s | 89 / 77 | 4.60 GiB |
| image caption through `Agent` (incl. load) | 18.2 s | 89 / 78 | 4.59 GiB |
| video caption `sample.mp4` through `Agent` (fps 4) | 43.2 s | 6,067 / 422 | 5.78 GiB |
| video caption (raw generate, processor default) | 15.3 s · 16.7 tok/s | 2,943 / 256 | 5.18 GiB |
| multi-turn follow-up | 18.1 s | 12,241 / 480 | 5.81 GiB |

## Generator — `Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Edge")`

Same Diffusers pipeline as Nano (`model_index.json` → `Cosmos3OmniPipeline`). The output field is
`video`: a flat list of PIL frames (one for text→image).

| Thor measurement (832×480, 8 steps, guidance 7) | time | peak GPU |
|---|---|---|
| pipeline load → cuda | 11.9–13.5 s | 7.61 GiB |
| text→image (1 frame) | 3.6–5.3 s | 9.02 GiB |
| text→video (17 frames) | 19.2–23.8 s (1.1–1.4 s/frame) | 12.09 GiB |

Sound: `cosmos3_text2video_sound` / `cosmos3_image2video_sound` return an error naming Nano as the
sound-capable checkpoint when `C3_MODEL`/`C3_GEN_MODEL` is Edge.

## Action — `cosmos3_action_edge` (no server, no Cosmos Framework)

```python
from strands_cosmos import cosmos3_action_edge
cosmos3_action_edge(image="sample.png", prompt="pick up the blue cube",
                    mode="policy", embodiment="droid_lerobot", chunk_size=16, steps=8)
# -> actions [16, 10] (already sliced to DROID's raw width), 17 frames, actions.json + frame_*.png
```

Modes: `policy` (actions + future video from one frame), `forward_dynamics` (video from
`raw_actions_json`), `inverse_dynamics` (actions between frames).

| Thor measurement (480 tier, chunk 16, 8 steps) | time | peak GPU |
|---|---|---|
| policy `droid_lerobot` (first call) | 60.2 s | 12.09 GiB |
| policy `bridge_orig_lerobot` (warm) | 35.6 s | 12.09 GiB |
| forward_dynamics `droid_lerobot`, 16×10 raw actions | 36.1 s | 12.09 GiB |

### Embodiments: only 10 of 32 slots are trained

Measured on the pinned revision: the per-domain output head (`action_proj_out`, 264,256 parameters per
slot) has a non-zero bias and weight-L2 5.4–13.4 for exactly these ids, and bias **exactly 0** for the
other 22. The trained set equals the ten embodiments NVIDIA's card lists.

| trained (`validate_embodiment` passes) | raw dim | registered but **untrained** (refused unless `allow_untrained=True`) |
|---|---|---|
| `av` 9 · `camera_pose` 9 · `hand_pose` 57 · `umi` 10 · `bridge_orig_lerobot` 10 · `droid_lerobot` / `robomind-franka` 10 · `robomind-franka-dual` 20 · `robomind-ur` 10 · `agibotworld` (+ `agibot_gear_gripper*`) 29 · `fractal` 10 | | `pusht` (2) · `libero` (no width — diffusers also refuses it) · `galbot` (30) · `no_action` |

`so101` / SO-100-class 6-DoF arms are **not registered**; zero-shot use is impossible and adding one means
post-training a new domain slot (see [Cosmos 3 training](cosmos3-training.md)).

## Where each surface actually runs

| surface | Nano | Edge |
|---|---|---|
| reason | `just c3-serve-reason` (vLLM, `.venv-c3-reason`) | `Cosmos3EdgeHFModel` in `.venv-c3-edge`, or vLLM |
| generate | `.venv-c3-gen` Diffusers | same, or `.venv-c3-edge` |
| action | Cosmos Framework torchrun (`../cosmos/packages/cosmos3`) | `cosmos3_action_edge` (Diffusers) or Framework |
| train | Cosmos Framework SFT (8×H100 recipes) | Cosmos Framework; see training guide for what runs on Thor |
