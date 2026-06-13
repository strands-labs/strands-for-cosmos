"""Cosmos 3 fine-tuning (SFT) — drive the Cosmos Framework training stack.

Supervised fine-tuning of Cosmos 3 via NVIDIA's cosmos-framework. Tested upstream
on 8x H100 (80GB). The convert / config-validation steps run on any GPU; the full
SFT run needs the documented multi-GPU allocation.

Prerequisites:
    just c3-setup-framework   # clones cosmos-framework + uv sync (cu130-train)

Full flow (vision SFT on Cosmos3-Nano):
    1. Convert base checkpoint -> DCP
    2. (optional) prepare your dataset JSONL
    3. Run SFT
    4. Export trained checkpoint -> HF safetensors

📓 Advanced example — no intro notebook. New here? Start with
   ../notebooks/00_start_here.ipynb and walk through 01–07 first.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands_cosmos import (
    cosmos3_train,
    cosmos3_train_convert,
    cosmos3_train_export,
    cosmos3_train_recipes,
    cosmos3_train_show,
)

# 0) Discover the SFT recipes shipped with your framework checkout.
print("=== Available SFT recipes ===")
print(cosmos3_train_recipes())

# 1) Inspect / validate a recipe's resolved config (no GPU needed).
print("\n=== Validate vision_sft_nano config (dry run) ===")
print(cosmos3_train_show(recipe="vision_sft_nano"))

# 2) Convert the base checkpoint to DCP (downloads + converts; ~minutes).
#    print(cosmos3_train_convert(checkpoint="Cosmos3-Nano"))

# 3) Run SFT (needs ~8 GPUs for the default recipe). Use overrides for a smoke:
#    print(cosmos3_train(
#        recipe="vision_sft_nano",
#        nproc=8,
#        overrides="trainer.max_iter=20 optimizer.lr=1e-5",
#    ))

# 4) Export the trained checkpoint to HF safetensors:
#    print(cosmos3_train_export(run_dir="outputs/train/cosmos3/sft/vision_sft_nano"))

print("\nSee docs/guide/cosmos3-training.md for the full multi-GPU launch.")
