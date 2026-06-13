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
