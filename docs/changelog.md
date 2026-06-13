# Changelog

All notable changes to **strands-cosmos** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.4] - 2026-06-04

### Changed

- **`cosmos3_video2video` now exposes the conditioning controls** that actually
  drive how much the prompt transforms the video:
  `condition_frames` (latent frame indexes kept clean) and `condition_keep`
  ("first"/"last"). New restyle-friendly defaults (`condition_frames="0"`,
  `condition_keep="last"`, `guidance=8`, `steps=35`) so a real transform
  (day→night, recolor, change the scene) is the easy path. With the old
  defaults the prompt barely changed the video (it reconstructed the input).
- `c3-v2v` justfile recipe gains matching `cond_frames` / `cond_keep` params.
- Added `demo/cosmos3_v2v/` showcase (recolor / rain / crowd / night) + tuning guide.

## [0.4.3] - 2026-06-04

### Added — native video-to-video transfer

- **`cosmos3_video2video` tool** + `c3-omni-docker` / `c3-v2v` justfile recipes:
  structure-preserving video transfer (e.g. day→night, recolor buildings,
  restyle) by re-rendering an input video with a new prompt via the Cosmos 3
  **vLLM-Omni** server (`/v1/videos/sync`, `input_reference`).
- vLLM-Omni runs from the official `vllm/vllm-omni:cosmos3` Docker image (the
  only build with all modalities incl. video2video). `c3-omni-docker` launches
  it with `--gpus all` and the shared HF cache.
- Validated live: input construction-site video transformed to night / recolor /
  rain (832×480, structure preserved, ~12s/clip).

## [0.4.2] - 2026-06-04

### Fixed

- **All justfile-backed tools failed on `pip install`**: the `justfile` (which
  every `cosmos_*` / `cosmos3_*` tool shells out to) was never packaged, so
  `_find_justfile()` returned `None` in a pip install and tools errored with
  exit 127. The justfile is now bundled into the wheel/sdist at
  `strands_cosmos/justfile` (copied from the repo root at build time) and the
  lookup resolves it. Verified from a clean wheel install. Added a regression test.

## [0.4.1] - 2026-06-04

### Fixed

- **PyPI publish failure (0.3.2 / 0.4.0 never reached PyPI)**: the `cosmos3-gen`
  and `all` extras declared a direct-URL dependency
  (`diffusers @ git+https://...`), which PyPI rejects on upload. The extras now
  require plain `diffusers>=0.36`; the Cosmos3OmniPipeline dev build is installed
  separately via `just c3-setup-gen` or
  `pip install -U "git+https://github.com/huggingface/diffusers.git"`.
- Release workflow now fails fast if built metadata contains any direct-URL
  dependency (guard step before PyPI publish).

## [0.4.0] - 2026-06-04

### Added — Cosmos 3 post-training (SFT)

- **7 `cosmos3_train_*` tools** + **`c3-train-*` justfile recipes** wrapping the
  NVIDIA Cosmos Framework supervised fine-tuning stack:
  `cosmos3_train_recipes`, `cosmos3_train_show`, `cosmos3_train_convert`,
  `cosmos3_train_convert_vlm`, `cosmos3_train_prep_dataset`, `cosmos3_train`,
  `cosmos3_train_export`.
- Full 4-step flow: checkpoint → DCP convert, dataset prep, SFT run (paired
  launch shell / `torchrun -m cosmos_framework.scripts.train`), and DCP → HF
  safetensors export. Recipes: vision_sft_nano/super, llava_ov, videophy2_nano.
- New **[Cosmos 3 Training guide](guide/cosmos3-training.md)** +
  `examples/10_cosmos3_finetune.py`; `c3-doctor` reports the training env.
- Validated locally (no 8× H100 needed): `c3-train-recipes` lists recipes and
  `c3-train-show` resolves/validates the full SFT config via `train.py --dryrun`.

## [0.3.2] - 2026-06-04

### Added

- **`cosmos3-gen` optional extra** — `pip install "strands-cosmos[cosmos3-gen]"`
  installs the Cosmos 3 generator backend in one step (Diffusers dev build,
  `cosmos_guardrail`, `soundfile`, imageio), making text/image -> image/video/sound
  generation frictionless without the justfile.

### Fixed

- `just c3-setup-gen` now also installs `soundfile` (required for the
  video-with-sound audio mux path).

## [0.3.1] - 2026-06-04

### Fixed

- **Video decoding on fresh installs**: `transformers` 5.x decodes video via
  `torchcodec` and silently falls back to `torchvision`, whose `io.read_video`
  was removed in `torchvision>=0.27` (`AttributeError` when captioning a video).
  `torchcodec` is now a core dependency so video works out of the box.

## [0.3.0] - 2026-06-04

### Added — Cosmos 3 omnimodal world models 🌌

First-class support for NVIDIA's newest [Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)
omnimodal model family — reasoning **and** generation across text, image, video, audio,
and action. Runs on local compute (vLLM / Diffusers / Cosmos Framework).

- **Model providers** (implement `strands.models.Model`):
  - `Cosmos3ReasonerModel` — omnimodal reasoning (text + vision → text) via a local
    vLLM server (`Cosmos3ReasonerForConditionalGeneration`). Captioning, temporal
    localization, embodied next-action, 2D grounding, physical plausibility, situation
    understanding, action chain-of-thought. Supports explicit `<think>` reasoning.
  - `Cosmos3GeneratorModel` — generation (text/image → image/video/**sound**) in-process
    via HuggingFace Diffusers `Cosmos3OmniPipeline`. Generates **and muxes stereo AAC
    audio @ 48kHz** for video-with-sound.
- **16 `cosmos3_*` tools** (thin justfile wrappers):
  - Reasoner: `cosmos3_reason`, `cosmos3_caption`, `cosmos3_temporal`, `cosmos3_embodied`,
    `cosmos3_ground`, `cosmos3_plausibility`, `cosmos3_situation`, `cosmos3_action_cot`
  - Generator: `cosmos3_text2image`, `cosmos3_text2video`, `cosmos3_image2video`,
    `cosmos3_text2video_sound`
  - Action / world-model: `cosmos3_forward_dynamics`, `cosmos3_inverse_dynamics`,
    `cosmos3_policy`
  - Servers: `cosmos3_serve`
- **13 `c3-*` justfile recipes**: `c3-doctor`, `c3-setup-{reason,gen,omni,framework}`,
  `c3-serve-{reason,omni,status,stop-reason,stop-omni}`, `c3-reason`, `c3-gen`, `c3-action`.
- **`cosmos3` optional extra** (`pip install "strands-cosmos[cosmos3]"`) for the reasoner
  OpenAI client.
- **Examples**: `06_cosmos3_reason.py`, `07_cosmos3_generate.py`, `08_cosmos3_action.py`,
  `09_cosmos3_showcase.py` (reason → generate showcase).
- **Showcase**: `demo/cosmos3_showcase/` — Cosmos 3 reasons about a real video, then
  generates similar videos (incl. one with synchronized audio) from its own description.
- **Docs**: new [Cosmos 3 Guide](guide/cosmos3.md); README, index, quickstart,
  installation, architecture, API reference, and capability guides all lead with Cosmos 3.

### Changed

- README and docs now present **4 model providers + 37 tools** and lead with Cosmos 3
  as the flagship; Cosmos-Reason2 repositioned as the lightweight edge/Jetson VLM.
- `c3-doctor` recognizes cached Hugging Face tokens (not just `HF_TOKEN`/env).
- `c3-serve-status` (and `cosmos3_serve`) detect directly-launched servers via an HTTP
  `/health` probe, not only the PID file.

### Verified (single NVIDIA L40S, 46 GB, no NIM)

- Reasoner: caption 6.6s, plus temporal / embodied / plausibility / situation.
- Generator: text→image, text→video, image→video, and text→video **+ sound**
  (H264 + AAC stereo 48kHz).
- Action: forward-dynamics rollout (832×480, 61 frames) via the Cosmos Framework.

### Notes

- **CUDA pairing**: match the torch backend to your driver — CUDA 13 → `cu130` +
  `vllm==0.21.0`; CUDA 12.8 → `cu128` + `vllm==0.19.1`. `just c3-doctor` reports it.
- **Single-GPU**: the reasoner (vLLM) and generator (Diffusers) each load a 16B model
  and will not fit on one ~46 GB GPU simultaneously — stop one before running the other.
- The Cosmos 3 reasoner caps `--max-model-len 32768` to avoid KV-cache OOM (the model's
  262K default exceeds a single 46 GB GPU).
- Removed the internal `COSMOS3_INTEGRATION.md` planning doc (superseded by shipped code + docs).

## [0.2.0] - 2026-05-08

### Added

- 21 tools covering the full Cosmos pipeline (Reason2 inference, Predict2.5,
  Transfer2.5, model lifecycle, training, distillation, Xenna curation, evaluation,
  I/O, system diagnostics) — all thin wrappers over a 50+ recipe `justfile`.
- MkDocs Material documentation site with examples, guides, and API reference.
- Notebooks and runnable examples (01–05).
- `AGENTS.md` development contract.

## [0.1.2] - 2026-03-xx

### Fixed

- asyncio event-loop thread-safety in `cosmos_vision_invoke`.

## [0.1.1] - 2026-03-09

### Added

- Initial public release: `CosmosVisionModel` (video + image + text) and `CosmosModel`
  (text-only) Strands model providers for Cosmos-Reason2.
- Jetson CUBLAS compatibility fix (`strands-cosmos-fix-cublas`).
- Dashcam safety-analysis demo.

[Unreleased]: https://github.com/cagataycali/strands-cosmos/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cagataycali/strands-cosmos/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cagataycali/strands-cosmos/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cagataycali/strands-cosmos/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cagataycali/strands-cosmos/releases/tag/v0.1.1
