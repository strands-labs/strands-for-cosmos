# Cosmos 3 — Post-Training (SFT)

Fine-tune Cosmos 3 models on your own data via NVIDIA's
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework) supervised
fine-tuning (SFT) stack. strands-cosmos exposes the framework's training flow as
`cosmos3_train_*` tools + `c3-train-*` justfile recipes (thin wrappers — the
framework is the single source of truth).

!!! warning "Hardware"
    Full SFT is tested upstream on **8× H100 (80 GB)**. The **convert**,
    **dataset-prep**, and **config-validation** steps run on any GPU (or none),
    so you can wire and validate the whole pipeline locally; the actual training
    run needs the documented multi-GPU allocation.

## Setup

```bash
just c3-setup-framework   # clone cosmos-framework -> ../cosmos/packages/cosmos3 + uv sync (cu130-train)
just c3-doctor            # confirms the training (SFT) env is present
```

## Recipes

```bash
just c3-train-recipes     # list SFT recipes + paired launch shells
```

| Recipe | Surface | Dataset | Base checkpoint |
|--------|---------|---------|-----------------|
| `vision_sft_nano` | Generator (T2V/I2V/V2V) | bridge-v2-subset-synthetic-captions | Cosmos3-Nano |
| `vision_sft_super` | Generator (LoRA, 64B) | bridge-v2-subset-synthetic-captions | Cosmos3-Super |
| `llava_ov` | Reasoner alignment | LLaVA-OneVision (HF stream) | Qwen3-VL-8B (fetched) |
| `videophy2_nano` | Reasoner alignment | VideoPhy-2 | Cosmos3-Nano-VLM |

## The 4-step flow

### 1. Convert the base checkpoint → DCP

```bash
just c3-train-convert Cosmos3-Nano             # -> examples/checkpoints/Cosmos3-Nano (DCP)
# Reasoner VLM path instead:
just c3-train-convert-vlm Cosmos3-Nano         # -> examples/checkpoints/Cosmos3-Nano-VLM
```

```python
from strands_cosmos import cosmos3_train_convert
cosmos3_train_convert(checkpoint="Cosmos3-Nano")
```

### 2. (optional) Prepare your dataset

Vision recipes expect a `train/video_dataset_file.jsonl`. Convert a captions
JSONL into the SFT format:

```bash
just c3-train-prep-dataset captions.jsonl sft_dataset.jsonl
```

```python
from strands_cosmos import cosmos3_train_prep_dataset
cosmos3_train_prep_dataset(captions="captions.jsonl", out="sft_dataset.jsonl")
```

### 3. Validate the config, then run SFT

Always dry-run first (no GPU) to confirm the resolved config:

```bash
just c3-train-show vision_sft_nano             # train.py --dryrun: prints the resolved config
```

```python
from strands_cosmos import cosmos3_train_show, cosmos3_train
cosmos3_train_show(recipe="vision_sft_nano")

# Full run (8 GPUs). Use Hydra tail overrides for short smokes / hyperparams:
cosmos3_train(
    recipe="vision_sft_nano",
    nproc=8,
    dataset="examples/data/.../sft_dataset_bridge",   # optional override
    checkpoint="examples/checkpoints/Cosmos3-Nano",   # the DCP from step 1
    overrides="trainer.max_iter=200 optimizer.lr=1e-5",
)
```

Under the hood this calls the framework's paired launch shell, which runs:

```bash
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
    --sft-toml=examples/toml/sft_config/vision_sft_nano.toml \
    -- trainer.max_iter=200 optimizer.lr=1e-5
```

### 4. Export the trained checkpoint → HF safetensors

```bash
just c3-train-export outputs/train/cosmos3/sft/vision_sft_nano
```

```python
from strands_cosmos import cosmos3_train_export
cosmos3_train_export(run_dir="outputs/train/cosmos3/sft/vision_sft_nano")
```

The exported safetensors can then be served back through `Cosmos3ReasonerModel`
(point a vLLM server at it) or loaded by `Cosmos3GeneratorModel`.

## Tools reference

| Tool | Purpose |
|------|---------|
| `cosmos3_train_recipes` | List SFT recipes + launch shells |
| `cosmos3_train_show` | Validate/print a recipe's resolved config (dry run) |
| `cosmos3_train_convert` | Base checkpoint → PyTorch DCP |
| `cosmos3_train_convert_vlm` | LM → Qwen3-VL visual tower (reasoner VLM) |
| `cosmos3_train_prep_dataset` | captions JSONL → SFT dataset JSONL |
| `cosmos3_train` | Run SFT via the paired launch shell |
| `cosmos3_train_export` | Trained DCP → HF safetensors |

> **Tip:** every recipe TOML defaults to `job.wandb_mode = "disabled"`. Set it to
> `"online"` and export `WANDB_API_KEY` to log a run to Weights & Biases.

See the [cosmos-framework training docs](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
for dataset licensing, OOM tuning, and the full Hydra override reference.

## Cosmos3-Edge on a single edge GPU (measured on Jetson AGX Thor)

The framework's recipes are written for 8×H100, but the *platform* is not the limit: on a Jetson AGX
Thor (arm64, CUDA sm_110, 128 GB unified memory) `just c3-setup-framework` (`uv sync --all-extras
--group=cu130-train`) completed in **58 s** (CPython 3.13, torch 2.10.0+cu130, CUDA on), and the
framework ships Edge recipes: `vision_sft_edge` (generator T2V/I2V/V2V) and `videophy2_sft_edge`
(reasoner), with `examples/launch_sft_vision_edge.sh` / `launch_sft_videophy2_edge.sh`.

| step | command | Thor result |
|---|---|---|
| 1 convert | `just c3-train-convert nvidia/Cosmos3-Edge` | **PASS** — 6.3 GB DCP in `examples/checkpoints/Cosmos3-Edge`, 70 s. The recipe resolves the HF id to the local snapshot (the framework only accepts a registered name or a directory). Needs the Wan2.2 VAE (`Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`, fetched automatically — do not set `HF_HUB_OFFLINE` unless it is cached). |
| 2 dataset | `train/video_dataset_file.jsonl` (see framework `docs/dataset_jsonl.md`) | `vision_path` may be an absolute local path. **Every window needs ≥ 61 frames** (`get_sft_dataset(min_frames=61)`), otherwise the loader ends with `AssertionError: Did not find any data` after the model has already been built. |
| 3 SFT smoke | `NPROC_PER_NODE=1 TAIL_OVERRIDES=(trainer.max_iter=10 checkpoint.save_iter=10 model.compile.enabled=false model.ema.enabled=false trainer.grad_accum_iter=1) bash examples/launch_sft_vision_edge.sh` | model build + DCP warm-start **PASS** on 1 GPU (checkpoint load 11.3 s, FSDP mesh [1,1], bf16, full activation checkpointing); GPU memory during the first step ≈ 119 GiB of 122.8 GiB — see the result row below. |
| 4 export + infer | `cosmos_framework.scripts.export_model` → `Cosmos3OmniPipeline` | see below |

Use the venv's `torchrun` (`.venv/bin`), not a system one — a `~/.local/bin/torchrun` first on PATH
launches `/usr/bin/python3` without the framework (`ModuleNotFoundError: omegaconf`).
