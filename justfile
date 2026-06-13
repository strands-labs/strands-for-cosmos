# strands-cosmos — Full NVIDIA Cosmos ecosystem for Strands Agents
# All tools shell out to `just <recipe>`; operators can run them directly.
# Recipes sourced from thor-cosmos; adapted for strands-cosmos standalone usage.

set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true
set positional-arguments := true

# Environment defaults
VENV              := ".venv"
PYTHON            := "python3"

# Cosmos repos (clone alongside strands-cosmos or set env vars)
COSMOS_PREDICT_REPO   := env_var_or_default("COSMOS_PREDICT_REPO", "../cosmos-predict2.5")
COSMOS_TRANSFER_REPO  := env_var_or_default("COSMOS_TRANSFER_REPO", "../cosmos-transfer2.5")
COSMOS_REASON_REPO    := env_var_or_default("COSMOS_REASON_REPO", "../cosmos-reason2")
COSMOS_XENNA_REPO     := env_var_or_default("COSMOS_XENNA_REPO", "../cosmos-xenna")
COSMOS_RL_REPO        := env_var_or_default("COSMOS_RL_REPO", "../cosmos-rl")
COSMOS_COOKBOOK_REPO   := env_var_or_default("COSMOS_COOKBOOK_REPO", "../cosmos-cookbook")

# TensorRT-Edge-LLM binaries (Thor-side)
TRT_ROOT              := env_var_or_default("TRT_ROOT", "/opt/tensorrt-edge-llm")
SERVER_BIN            := env_var_or_default("COSMOS_SERVER_BIN", TRT_ROOT + "/build/examples/server/trt_edgellm_server")
LLM_BUILD_BIN         := env_var_or_default("TRT_LLM_BUILD_BIN", TRT_ROOT + "/build/examples/llm/llm_build")
VISUAL_BUILD_BIN      := env_var_or_default("TRT_VISUAL_BUILD_BIN", TRT_ROOT + "/build/examples/multimodal/visual_build")

# Serve config
VLM_HOST              := env_var_or_default("VLM_HOST", "127.0.0.1")
VLM_PORT              := env_var_or_default("VLM_PORT", "8080")
VLM_URL               := "http://" + VLM_HOST + ":" + VLM_PORT + "/v1/chat/completions"

# RTP / NATS
RTP_BIND              := env_var_or_default("RTP_BIND", "0.0.0.0")
RTP_PORT              := env_var_or_default("RTP_PORT", "5600")
NATS_URL              := env_var_or_default("NATS_URL", "nats://127.0.0.1:4222")

PID_FILE              := env_var_or_default("COSMOS_SERVER_PID", "/tmp/strands-cosmos-server.pid")
LOG_FILE              := env_var_or_default("COSMOS_SERVER_LOG", "/tmp/strands-cosmos-server.log")


# Git URLs for auto-clone
# Override with env vars if you use forks or SSH URLs
COSMOS_PREDICT_GIT    := env_var_or_default("COSMOS_PREDICT_GIT", "https://github.com/nvidia-cosmos/cosmos-predict2.5.git")
COSMOS_TRANSFER_GIT   := env_var_or_default("COSMOS_TRANSFER_GIT", "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git")
COSMOS_REASON_GIT     := env_var_or_default("COSMOS_REASON_GIT", "https://github.com/nvidia-cosmos/cosmos-reason2.git")
COSMOS_XENNA_GIT      := env_var_or_default("COSMOS_XENNA_GIT", "https://github.com/nvidia-cosmos/cosmos-curate.git")
COSMOS_RL_GIT         := env_var_or_default("COSMOS_RL_GIT", "https://github.com/nvidia-cosmos/cosmos-rl.git")
COSMOS_COOKBOOK_GIT    := env_var_or_default("COSMOS_COOKBOOK_GIT", "https://github.com/nvidia-cosmos/cosmos-cookbook.git")


# Top-level
default:
    @just --list --unsorted

# Print the effective environment
env:
    @echo "COSMOS_PREDICT_REPO  = {{COSMOS_PREDICT_REPO}}"
    @echo "COSMOS_TRANSFER_REPO = {{COSMOS_TRANSFER_REPO}}"
    @echo "COSMOS_REASON_REPO   = {{COSMOS_REASON_REPO}}"
    @echo "COSMOS_XENNA_REPO    = {{COSMOS_XENNA_REPO}}"
    @echo "COSMOS_RL_REPO       = {{COSMOS_RL_REPO}}"
    @echo "COSMOS_COOKBOOK_REPO  = {{COSMOS_COOKBOOK_REPO}}"
    @echo "TRT_ROOT             = {{TRT_ROOT}}"
    @echo "VLM_URL              = {{VLM_URL}}"
    @echo "NATS_URL             = {{NATS_URL}}"


# Auto-clone / ensure repos
# `just setup` clones all missing repos. Individual `ensure-*` recipes are
# called as deps by recipes that need a specific repo.

# Clone a repo if the target dir doesn't exist
[private]
_clone url dir:
    #!/usr/bin/env bash
    if [ -d "{{dir}}" ]; then
      echo "✓ {{dir}} exists"
    else
      echo "📥 cloning {{url}} → {{dir}}"
      git clone --depth 1 "{{url}}" "{{dir}}"
    fi

# Ensure individual repos exist (call before recipes that need them)
ensure-predict:
    @just _clone "{{COSMOS_PREDICT_GIT}}" "{{COSMOS_PREDICT_REPO}}"

ensure-transfer:
    @just _clone "{{COSMOS_TRANSFER_GIT}}" "{{COSMOS_TRANSFER_REPO}}"

ensure-reason:
    @just _clone "{{COSMOS_REASON_GIT}}" "{{COSMOS_REASON_REPO}}"

ensure-xenna:
    @just _clone "{{COSMOS_XENNA_GIT}}" "{{COSMOS_XENNA_REPO}}"

ensure-rl:
    @just _clone "{{COSMOS_RL_GIT}}" "{{COSMOS_RL_REPO}}"

ensure-cookbook:
    @just _clone "{{COSMOS_COOKBOOK_GIT}}" "{{COSMOS_COOKBOOK_REPO}}"

# Clone ALL repos (one-shot dev setup)
setup: ensure-predict ensure-transfer ensure-reason ensure-xenna ensure-rl ensure-cookbook
    @echo ""
    @echo "✅ All Cosmos repos cloned. Running doctor..."
    @echo ""
    @just doctor

# Full setup: system deps + python + repos + TRT (takes ~30min on Jetson)
setup-full:
    just install-system-deps
    just install-python-deps
    just setup
    @echo ""
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @echo "🎯 Next steps:"
    @echo "  For edge VLM (Jetson Thor):"
    @echo "    just install-trt-edge-llm"
    @echo "    just pipeline-edge-deploy"
    @echo ""
    @echo "  For HF inference (any GPU):"
    @echo "    just download reason2-2b"
    @echo "    python -c \"from strands_cosmos import CosmosVisionModel; print('ok')\""
    @echo ""

# Pull latest on all repos (useful after setup)
update:
    #!/usr/bin/env bash
    for dir in "{{COSMOS_PREDICT_REPO}}" "{{COSMOS_TRANSFER_REPO}}" "{{COSMOS_REASON_REPO}}" \
               "{{COSMOS_XENNA_REPO}}" "{{COSMOS_RL_REPO}}" "{{COSMOS_COOKBOOK_REPO}}"; do
      if [ -d "$dir/.git" ]; then
        echo "⬆ pulling $dir"
        git -C "$dir" pull --rebase 2>/dev/null || git -C "$dir" pull || true
      fi
    done

# Check which repos are present/missing
doctor:
    #!/usr/bin/env bash
    echo "🩺 strands-cosmos doctor"
    echo "========================"
    echo ""
    echo "📦 Cosmos Repos:"
    check() {
      if [ -d "$2" ]; then
        echo "  ✅ $1 → $2"
      else
        echo "  ❌ $1 → $2 (MISSING — run 'just setup' or 'just ensure-$3')"
      fi
    }
    check "Predict 2.5"  "{{COSMOS_PREDICT_REPO}}"  "predict"
    check "Transfer 2.5" "{{COSMOS_TRANSFER_REPO}}" "transfer"
    check "Reason 2"     "{{COSMOS_REASON_REPO}}"   "reason"
    check "Xenna/Curate" "{{COSMOS_XENNA_REPO}}"    "xenna"
    check "RL"           "{{COSMOS_RL_REPO}}"        "rl"
    check "Cookbook"      "{{COSMOS_COOKBOOK_REPO}}"  "cookbook"
    echo ""

    # Detect platform
    ARCH=$(uname -m)
    IS_JETSON=false
    [ -f /proc/device-tree/model ] && IS_JETSON=true
    IS_DOCKER=false
    [ -f /.dockerenv ] && IS_DOCKER=true

    echo "🖥  Platform:"
    echo "  arch: $ARCH"
    if $IS_JETSON; then
      echo "  type: 🟢 Jetson ($(cat /proc/device-tree/model 2>/dev/null | tr -d '\0'))"
    elif $IS_DOCKER; then
      echo "  type: 🐳 Docker container"
    else
      echo "  type: 🖥  Workstation / Cloud"
    fi
    echo ""

    # Core tools (needed everywhere)
    echo "🔧 Core Tools (needed on all platforms):"
    for bin in python3 pip git just hf curl jq; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin ($(command -v $bin))"
      else
        echo "  ❌ $bin — REQUIRED"
      fi
    done
    echo ""

    # Python packages
    echo "🐍 Python Packages:"
    for pkg in strands_agents strands_cosmos torch transformers accelerate; do
      if python3 -c "import $pkg" 2>/dev/null; then
        VER=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
        echo "  ✅ $pkg ($VER)"
      else
        echo "  ⚠️  $pkg (not installed)"
      fi
    done
    echo ""

    # Media / I/O tools (optional but useful)
    echo "📹 Media & I/O (optional):"
    for bin in ffmpeg ffprobe gst-launch-1.0 nats; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin"
      else
        echo "  ⚠️  $bin (not found — some tools will be limited)"
      fi
    done
    echo ""

    # TensorRT / Edge tools (only on Jetson or TRT docker)
    echo "⚡ TensorRT-Edge-LLM (Jetson or TRT Docker only):"
    TRT_TOOLS=(tensorrt-edgellm-quantize-llm tensorrt-edgellm-export-llm tensorrt-edgellm-export-visual)
    TRT_FOUND=0
    for bin in "${TRT_TOOLS[@]}"; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin"
        TRT_FOUND=$((TRT_FOUND + 1))
      else
        echo "  ⬜ $bin (not found)"
      fi
    done
    if [ -x "{{SERVER_BIN}}" ]; then
      echo "  ✅ trt_edgellm_server → {{SERVER_BIN}}"
      TRT_FOUND=$((TRT_FOUND + 1))
    else
      echo "  ⬜ trt_edgellm_server → {{SERVER_BIN}} (not found)"
    fi
    if [ -x "{{LLM_BUILD_BIN}}" ]; then
      echo "  ✅ llm_build → {{LLM_BUILD_BIN}}"
    else
      echo "  ⬜ llm_build → {{LLM_BUILD_BIN}} (not found)"
    fi
    if [ -x "{{VISUAL_BUILD_BIN}}" ]; then
      echo "  ✅ visual_build → {{VISUAL_BUILD_BIN}}"
    else
      echo "  ⬜ visual_build → {{VISUAL_BUILD_BIN}} (not found)"
    fi
    echo ""
    if [ $TRT_FOUND -eq 0 ]; then
      if $IS_JETSON; then
        echo "  ⚠️  TRT tools missing on Jetson! Install with:"
        echo "     just install-trt-edge-llm"
        echo "     (builds from source ~30min, or set TRT_ROOT if already built)"
      else
        echo "  ℹ️  TRT tools not found — EXPECTED on workstation."
        echo "     Quantize/export runs in the TRT docker container or on Jetson."
        echo "     Engine build + serve runs on Jetson Thor."
        echo "     Install: just install-trt-edge-llm /path/to/trt"
        echo "     Recipes: quantize, export-llm, export-visual, build-*-engine, serve-*"
        echo "     These will return exit 127 without TRT installed — that's normal."
      fi
    fi
    echo ""

    # CUDA
    echo "🎮 GPU/CUDA:"
    if command -v nvidia-smi &>/dev/null; then
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | while read line; do
        echo "  ✅ $line"
      done
    else
      echo "  ⬜ nvidia-smi not found (no GPU or driver not in PATH)"
    fi
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      DEVICE=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
      echo "  ✅ torch.cuda → $DEVICE"
    else
      echo "  ⚠️  torch.cuda not available (CPU-only — VLM inference will be slow)"
    fi
    echo ""

    # Summary
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 What works on THIS machine:"
    echo "  • cosmos_reason_hf / cosmos_vision_invoke (HF direct inference)"
    echo "  • cosmos_model_download (HF downloads)"
    echo "  • video_probe / video_extract_frames / image_read"
    echo "  • cosmos_evaluate (if cookbook cloned)"
    echo "  • cosmos_curate (if xenna cloned + deps)"
    echo "  • cosmos_post_train (reason2 SFT/LoRA if GPU available)"
    if [ $TRT_FOUND -gt 0 ] || $IS_JETSON; then
      echo "  • cosmos_quantize / export / build_engine / serve (TRT available)"
      echo "  • cosmos_inference (against local TRT server)"
      echo "  • rtp_capture_frame (GStreamer RTP)"
    else
      echo ""
      echo "  ⚡ For TRT edge pipeline (quantize → export → build → serve):"
      echo "     Run on Jetson Thor or inside TRT docker."
      echo "     Recipes: just quantize, just export-llm, just build-engines, just serve-start"
    fi
    echo ""

# Install
install:
    {{PYTHON}} -m venv {{VENV}} || true
    {{VENV}}/bin/pip install -U pip
    {{VENV}}/bin/pip install -e .
    @echo "✅ installed. Try: just smoke"

# Install TensorRT-Edge-LLM (builds from source — Jetson or x86 with CUDA)
# This provides: trt_edgellm_server, llm_build, visual_build,
#                tensorrt-edgellm-quantize-llm, tensorrt-edgellm-export-llm, etc.
install-trt-edge-llm trt_dir=TRT_ROOT:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "⚡ Installing TensorRT-Edge-LLM → {{trt_dir}}"
    echo ""

    # Prerequisites check
    for bin in cmake g++ ninja-build git; do
      if ! command -v "$bin" &>/dev/null; then
        echo "❌ $bin not found. Install build deps first:"
        echo "   sudo apt-get install -y cmake g++ ninja-build git python3-dev"
        exit 1
      fi
    done

    # Check TensorRT is available (JetPack provides it)
    if ! pkg-config --exists nvinfer 2>/dev/null && [ ! -f /usr/lib/aarch64-linux-gnu/libnvinfer.so ]; then
      echo "❌ TensorRT not found. On Jetson, ensure JetPack is installed."
      echo "   dpkg -l | grep libnvinfer"
      exit 1
    fi

    # Clone if not present
    if [ ! -d "{{trt_dir}}" ]; then
      echo "📥 Cloning TensorRT-Edge-LLM..."
      git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "{{trt_dir}}"
    else
      echo "✓ {{trt_dir}} exists"
    fi

    cd "{{trt_dir}}"

    # Build
    echo "🔨 Building (this may take 20-60 min on Jetson)..."
    mkdir -p build && cd build
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="87;90;100" \
      -DBUILD_EXAMPLES=ON \
      2>&1 | tail -5
    ninja -j$(nproc) 2>&1 | tail -20

    echo ""
    echo "✅ Build complete!"
    echo ""

    # Verify binaries exist
    for bin in examples/server/trt_edgellm_server examples/llm/llm_build examples/multimodal/visual_build; do
      if [ -x "build/$bin" ] || [ -x "$bin" ]; then
        echo "  ✅ $bin"
      else
        echo "  ⚠️  $bin not found (build may use different paths)"
      fi
    done

    # Install python tools if available
    if [ -f setup.py ] || [ -f pyproject.toml ]; then
      echo ""
      echo "📦 Installing Python tools (quantize, export)..."
      pip3 install -e . 2>&1 | tail -5 || true
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Set TRT_ROOT to use with strands-cosmos:"
    echo "  export TRT_ROOT={{trt_dir}}"
    echo "  # or add to .env:"
    echo "  echo 'TRT_ROOT={{trt_dir}}' >> .env"

# Install system deps (apt packages needed for build + runtime)
install-system-deps:
    #!/usr/bin/env bash
    echo "📦 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
      cmake g++ ninja-build git python3-dev \
      ffmpeg gstreamer1.0-tools gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-libav \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      jq curl
    echo ""
    echo "✅ System deps installed."
    echo "   For NATS: curl -sf https://binaries.nats.dev/nats-io/natscli/nats@latest | sh"

# Install Python deps (strands-cosmos + all extras)
install-python-deps:
    #!/usr/bin/env bash
    echo "🐍 Installing Python dependencies..."
    pip3 install -U pip
    pip3 install -e ".[all]"
    pip3 install strands-agents strands-agents-tools
    echo "✅ Python deps installed."

# Model / dataset download
download name="reason2-2b" local_dir="":
    #!/usr/bin/env bash
    DEST="{{local_dir}}"
    [ -z "$DEST" ] && DEST="./checkpoints/{{name}}"
    mkdir -p "$DEST"
    case "{{name}}" in
      reason2-2b)        REPO="nvidia/Cosmos-Reason2-2B" ;;
      reason2-7b)        REPO="nvidia/Cosmos-Reason2-7B" ;;
      reason1-7b-reward) REPO="nvidia/Cosmos-Reason1-7B-Reward" ;;
      predict2.5-2b)     REPO="nvidia/Cosmos-Predict2.5-2B" ;;
      predict2.5-14b)    REPO="nvidia/Cosmos-Predict2.5-14B" ;;
      transfer2.5-2b)    REPO="nvidia/Cosmos-Transfer2.5-2B" ;;
      transfer2.5-edge)  REPO="nvidia/Cosmos-Transfer2.5-Edge" ;;
      transfer2.5-depth) REPO="nvidia/Cosmos-Transfer2.5-Depth" ;;
      transfer2.5-seg)   REPO="nvidia/Cosmos-Transfer2.5-Seg" ;;
      *)                 REPO="{{name}}" ;;
    esac
    hf download "$REPO" --local-dir "$DEST"

download-dataset name="gr1" local_dir="":
    #!/usr/bin/env bash
    DEST="{{local_dir}}"
    [ -z "$DEST" ] && DEST="./datasets/{{name}}"
    mkdir -p "$DEST"
    case "{{name}}" in
      gr1)         REPO="nvidia/PhysicalAI-Robotics-GR00T-GR1" ;;
      gr1-100)     REPO="nvidia/GR1-100" ;;
      gr00t-eval)  REPO="nvidia/PhysicalAI-Robotics-GR00T-Eval" ;;
      safe-unsafe) REPO="pjramg/Safe_Unsafe_Test" ;;
      *)           REPO="{{name}}" ;;
    esac
    hf download "$REPO" --repo-type dataset --local-dir "$DEST"


# Quantization + ONNX export (x86 GPU host)
quantize model_dir="nvidia/Cosmos-Reason2-2B" output_dir="./quantized/Cosmos-Reason2-2B-fp8" dtype="fp16" quantization="fp8":
    mkdir -p "{{output_dir}}"
    tensorrt-edgellm-quantize-llm \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}" \
      --dtype "{{dtype}}" \
      --quantization "{{quantization}}"

export-llm model_dir output_dir:
    mkdir -p "{{output_dir}}"
    tensorrt-edgellm-export-llm \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}"

export-visual model_dir output_dir dtype="fp16" quantization="":
    mkdir -p "{{output_dir}}"
    #!/usr/bin/env bash
    CMD=(tensorrt-edgellm-export-visual \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}" \
      --dtype "{{dtype}}")
    [ -n "{{quantization}}" ] && CMD+=(--quantization "{{quantization}}")
    "${CMD[@]}"

prep-edge-model model="reason2-2b" out_root="./models/Cosmos-Reason2-2B-fp8":
    just download "{{model}}" "{{out_root}}/hf"
    just quantize "{{out_root}}/hf" "{{out_root}}/quantized" fp16 fp8
    just export-llm "{{out_root}}/quantized" "{{out_root}}/onnx"
    just export-visual "{{out_root}}/hf" "{{out_root}}/onnx/visual_enc_onnx" fp16 fp8
    @echo "✅ ONNX ready → {{out_root}}/onnx  (scp to Thor next)"


# TRT engine build (on Thor)
build-llm-engine onnx_dir engine_dir min_tokens="4" max_tokens="10240" max_input_len="1024":
    mkdir -p "{{engine_dir}}"
    "{{LLM_BUILD_BIN}}" \
      --onnxDir "{{onnx_dir}}" \
      --engineDir "{{engine_dir}}" \
      --vlm \
      --minImageTokens {{min_tokens}} \
      --maxImageTokens {{max_tokens}} \
      --maxInputLen {{max_input_len}}

build-visual-engine onnx_dir engine_dir:
    mkdir -p "{{engine_dir}}"
    "{{VISUAL_BUILD_BIN}}" \
      --onnxDir "{{onnx_dir}}" \
      --engineDir "{{engine_dir}}"

build-engines onnx_dir engine_root:
    just build-llm-engine    "{{onnx_dir}}" "{{engine_root}}/llm"
    just build-visual-engine "{{onnx_dir}}/visual_enc_onnx" "{{engine_root}}/visual"


# Inference server (on Thor)
serve-start llm_engine_dir visual_engine_dir port=VLM_PORT host=VLM_HOST:
    #!/usr/bin/env bash
    if [ -f "{{PID_FILE}}" ] && kill -0 "$(cat {{PID_FILE}})" 2>/dev/null; then
      echo "🟢 already running (pid=$(cat {{PID_FILE}}))"; exit 0
    fi
    nohup "{{SERVER_BIN}}" \
      --llmEngineDir "{{llm_engine_dir}}" \
      --visualEngineDir "{{visual_engine_dir}}" \
      --host "{{host}}" --port "{{port}}" \
      >> "{{LOG_FILE}}" 2>&1 &
    echo $! > "{{PID_FILE}}"
    sleep 1
    echo "▶ started pid=$(cat {{PID_FILE}})  http://{{host}}:{{port}}"

serve-stop:
    #!/usr/bin/env bash
    if [ ! -f "{{PID_FILE}}" ]; then echo "🔴 not running"; exit 0; fi
    PID=$(cat "{{PID_FILE}}")
    if kill -0 "$PID" 2>/dev/null; then kill "$PID" && echo "⏹ stopped pid=$PID"; fi
    rm -f "{{PID_FILE}}"

serve-status:
    #!/usr/bin/env bash
    if [ -f "{{PID_FILE}}" ] && kill -0 "$(cat {{PID_FILE}})" 2>/dev/null; then
      echo "🟢 running pid=$(cat {{PID_FILE}})  {{VLM_URL}}"
    else
      echo "🔴 not running"; rm -f "{{PID_FILE}}"
    fi

serve-logs lines="80":
    @tail -n {{lines}} "{{LOG_FILE}}" 2>/dev/null || echo "no log yet"

serve-restart llm_engine_dir visual_engine_dir:
    -just serve-stop
    just serve-start "{{llm_engine_dir}}" "{{visual_engine_dir}}"


# Inference (HTTP)
infer image prompt="describe the scene" max_tokens="256" temperature="0.2" url=VLM_URL:
    #!/usr/bin/env bash
    # image/prompt are untrusted -> passed via env (C3_INFER_IMAGE/
    # C3_INFER_PROMPT), NOT {param}-interpolated into bash/JSON (CWE-78).
    # Routes through the hardened cosmos_inference tool (SSRF + workspace
    # confine), the SAME code path the agent uses -- no duplicated curl/jq.
    set -euo pipefail
    C3_INFER_IMAGE="{{image}}" C3_INFER_PROMPT="{{prompt}}" C3_INFER_URL="{{url}}" \
    C3_INFER_MAX_TOKENS="{{max_tokens}}" C3_INFER_TEMP="{{temperature}}" \
      python3 -m strands_cosmos.scripts.c3_cli infer


# RTP capture (GStreamer)
rtp-capture port=RTP_PORT width="800" height="600" timeout_s="5":
    #!/usr/bin/env bash
    # the output path is LLM-controlled -> read from $RTP_OUTPUT env
    # (no {param} interpolation -> no recipe breakout, CWE-78). Numerics stay
    # positional and are validated as ints by the calling tool.
    set -euo pipefail
    OUT="${RTP_OUTPUT:-/tmp/cosmos_frame.jpg}"
    timeout {{timeout_s}} gst-launch-1.0 -e \
      udpsrc address="${RTP_BIND:-0.0.0.0}" port={{port}} \
        caps='application/x-rtp,media=video,encoding-name=H264,payload=96' ! \
      rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! \
      video/x-raw,width={{width}},height={{height}},format=I420 ! \
      nvjpegenc ! filesink location="$OUT" || \
    timeout {{timeout_s}} gst-launch-1.0 -e \
      udpsrc address="${RTP_BIND:-0.0.0.0}" port={{port}} \
        caps='application/x-rtp,media=video,encoding-name=H264,payload=96' ! \
      rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! \
      video/x-raw,width={{width}},height={{height}} ! jpegenc ! \
      filesink location="$OUT"
    ls -la "$OUT"


# NATS publish
nats-publish subject payload_json:
    #!/usr/bin/env bash
    echo '{{payload_json}}' | nats pub "{{subject}}" --server "{{NATS_URL}}"


# Generation (Predict 2.5 / Transfer 2.5)
predict-generate input_json repo=COSMOS_PREDICT_REPO: ensure-predict
    cd "{{repo}}" && just run python examples/inference.py -i "{{input_json}}"

transfer-generate input_json control="edge" repo=COSMOS_TRANSFER_REPO: ensure-transfer
    cd "{{repo}}" && just run python examples/inference.py -i "{{input_json}}" "{{control}}"


# Post-training
post-train-reason2 config strategy="full":
    cosmos-cli train --config "{{config}}" --strategy "{{strategy}}"

post-train-reason2-rl config: ensure-rl
    cosmos-rl --config "{{config}}"

post-train-predict config num_gpus="8" repo=COSMOS_PREDICT_REPO: ensure-predict
    cd "{{repo}}" && torchrun --nproc-per-node={{num_gpus}} -m cosmos_predict2.train --config "{{config}}"

post-train-transfer config num_gpus="8" repo=COSMOS_TRANSFER_REPO: ensure-transfer
    cd "{{repo}}" && torchrun --nproc-per-node={{num_gpus}} -m cosmos_transfer2.train --config "{{config}}"


# Distillation
distill teacher student method="kd" family="transfer2_5" num_gpus="8":
    #!/usr/bin/env bash
    MODULE="cosmos_transfer2.distill"
    [ "{{family}}" = "predict2_5" ] && MODULE="cosmos_predict2.distill"
    torchrun --nproc-per-node={{num_gpus}} -m "$MODULE" \
      --method "{{method}}" \
      --teacher-ckpt "{{teacher}}" \
      --student-output "{{student}}"


# Data curation (Cosmos-Xenna)
curate input_dir output_dir="./outputs/curated" stages="all" workers="8" repo=COSMOS_XENNA_REPO: ensure-xenna
    cd "{{repo}}" && just run python -m cosmos_xenna.pipelines.v1.curate \
      --input-dir "{{input_dir}}" --output-dir "{{output_dir}}" \
      --stages "{{stages}}" --workers {{workers}}


# Evaluation
evaluate metric pred gt="" output_dir="./outputs/eval" repo=COSMOS_COOKBOOK_REPO: ensure-cookbook
    #!/usr/bin/env bash
    declare -A MAP=(
      [fid]=scripts/metrics/qualitative/compute_fid.py
      [fvd]=scripts/metrics/qualitative/compute_fvd.py
      [tse]=scripts/metrics/geometrical_consistency/compute_tse.py
      [cse]=scripts/metrics/geometrical_consistency/compute_cse.py
      [sampson]=scripts/metrics/geometrical_consistency/compute_sampson.py
      [blur_ssim]=scripts/metrics/control/compute_blur_ssim.py
      [canny_f1]=scripts/metrics/control/compute_canny_f1.py
      [depth_rmse]=scripts/metrics/control/compute_depth_rmse.py
      [seg_miou]=scripts/metrics/control/compute_seg_miou.py
      [dover]=scripts/metrics/control/compute_dover.py
      [reason_critic]=scripts/evaluation/reason_critic.py
      [reason_reward]=scripts/evaluation/cosmos-reason1-reward-7b/run.py
    )
    SCRIPT="${MAP[{{metric}}]}"
    if [ -z "$SCRIPT" ]; then echo "unknown metric: {{metric}}"; exit 2; fi
    mkdir -p "{{output_dir}}"
    CMD=(python "$SCRIPT" --pred "{{pred}}" --output "{{output_dir}}")
    [ -n "{{gt}}" ] && CMD+=(--gt "{{gt}}")
    cd "{{repo}}" && "${CMD[@]}"


# Video / image utils
video-probe video:
    ffprobe -v error -print_format json -show_format -show_streams "{{video}}"

video-frames video output_dir="/tmp/frames" fps="1.0" max_frames="0":
    #!/usr/bin/env bash
    mkdir -p "{{output_dir}}"
    CMD=(ffmpeg -y -hide_banner -loglevel warning -i "{{video}}" -vf fps={{fps}})
    [ "{{max_frames}}" != "0" ] && CMD+=(-frames:v {{max_frames}})
    CMD+=("{{output_dir}}/frame_%06d.jpg")
    "${CMD[@]}" 2>&1 || true
    ls "{{output_dir}}" | head -5


# System diagnostics
sysinfo:
    @echo "--- host ---"
    @hostname && uname -a
    @echo "--- jetson ---"
    @cat /proc/device-tree/model 2>/dev/null || echo "not a Jetson"
    @echo "--- nvidia-smi ---"
    @nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
    @echo "--- memory ---"
    @free -h 2>/dev/null || vm_stat
    @echo "--- thermal ---"
    @for z in /sys/class/thermal/thermal_zone*; do [ -r $z/temp ] && echo "$(cat $z/type 2>/dev/null || basename $z): $(awk '{printf "%.1fC\n", $1/1000}' $z/temp)"; done 2>/dev/null || true


# Pipelines (end-to-end)
pipeline-edge-deploy model="reason2-2b" out_root="./models/Cosmos-Reason2-2B-fp8":
    @echo "🏗  prep on x86 host"
    just prep-edge-model "{{model}}" "{{out_root}}"
    @echo "📤  scp ONNX to Thor, then:"
    @echo "🔨  just build-engines {{out_root}}/onnx {{out_root}}/engines"
    @echo "▶️  just serve-start {{out_root}}/engines/llm {{out_root}}/engines/visual"

pipeline-gr00t-dreams dataset_dir="./datasets/gr1" config="configs/gr00t-dreams.yaml":
    just download-dataset gr1 "{{dataset_dir}}"
    just post-train-predict "{{config}}"

perception-loop subject="perception.vlm" prompt="Describe the scene; count people.":
    #!/usr/bin/env bash
    echo "🔁 perception-loop starting. Ctrl-C to stop."
    while true; do
      FRAME=/tmp/cosmos_perception.jpg
      just rtp-capture {{RTP_PORT}} "$FRAME" 800 600 5 >/dev/null || { sleep 1; continue; }
      RESULT=$(just infer "$FRAME" "{{prompt}}" 128 0.1 2>/dev/null || echo "")
      [ -z "$RESULT" ] && { sleep 1; continue; }
      PAYLOAD=$(printf '{"text":%s,"ts":%d}' "$(printf '%s' "$RESULT" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()))')" "$(date +%s)")
      just nats-publish "{{subject}}" "$PAYLOAD" || true
      sleep 0.1
    done

# Smoke test
smoke:
    just env
    just sysinfo
    -just serve-status

# Development
test:
    {{PYTHON}} -m pytest -v tests/

lint:
    {{PYTHON}} -m ruff check strands_cosmos/

format:
    {{PYTHON}} -m ruff format strands_cosmos/


# ══════════════════════════════════════════════════════════════════════════
# Cosmos 3 — omnimodal world models (Reasoner + Generator + Action)
# No NIM. Local compute. uv-managed venvs. cu130 to match CUDA 13 driver.
# Branch: feat/cosmos3-integration
# ══════════════════════════════════════════════════════════════════════════

# Cosmos 3 config
C3_REPO            := env_var_or_default("C3_REPO", "../cosmos")
C3_FRAMEWORK_REPO  := env_var_or_default("C3_FRAMEWORK_REPO", "../cosmos/packages/cosmos3")
C3_MODEL           := env_var_or_default("C3_MODEL", "nvidia/Cosmos3-Nano")
C3_TORCH_BACKEND   := env_var_or_default("C3_TORCH_BACKEND", "cu130")
C3_VLLM_VERSION    := env_var_or_default("C3_VLLM_VERSION", "0.21.0")
C3_REASON_VENV     := env_var_or_default("C3_REASON_VENV", ".venv-c3-reason")
C3_GEN_VENV        := env_var_or_default("C3_GEN_VENV", ".venv-c3-gen")
C3_REASON_PORT     := env_var_or_default("C3_REASON_PORT", "8000")
C3_OMNI_PORT       := env_var_or_default("C3_OMNI_PORT", "8001")
C3_REASON_PID      := env_var_or_default("C3_REASON_PID", "/tmp/c3-reason-server.pid")
C3_REASON_LOG      := env_var_or_default("C3_REASON_LOG", "/tmp/c3-reason-server.log")
C3_OMNI_PID        := env_var_or_default("C3_OMNI_PID", "/tmp/c3-omni-server.pid")
C3_OMNI_LOG        := env_var_or_default("C3_OMNI_LOG", "/tmp/c3-omni-server.log")

# Cosmos 3 environment doctor
c3-doctor:
    #!/usr/bin/env bash
    echo "=== Cosmos 3 Doctor ==="
    echo "-- GPU --"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
    echo "-- Driver CUDA --"
    nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" || true
    echo "-- torch-backend pairing --  (driver CUDA 13 -> cu130 + vllm==0.21.0; CUDA 12.8 -> cu128 + vllm==0.19.1)"
    echo "  C3_TORCH_BACKEND={{C3_TORCH_BACKEND}}  C3_VLLM_VERSION={{C3_VLLM_VERSION}}"
    echo "-- uv --"
    command -v uv >/dev/null && uv --version || echo "uv NOT installed (https://docs.astral.sh/uv/)"
    echo "-- HF auth --"
    if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
      echo "  authenticated (env token)"
    elif [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ] || [ -f "$HOME/.cache/huggingface/token" ]; then
      echo "  authenticated (cached token)"
    else
      echo "  NOT authenticated; run: uvx hf@latest auth login (or export HF_TOKEN=...)"
    fi
    echo "-- Reasoner venv ({{C3_REASON_VENV}}) --"
    [ -d "{{C3_REASON_VENV}}" ] && echo "  present" || echo "  missing -> just c3-setup-reason"
    echo "-- Generator venv ({{C3_GEN_VENV}}) --"
    [ -d "{{C3_GEN_VENV}}" ] && echo "  present" || echo "  missing -> just c3-setup-gen"
    echo "-- Cosmos repo ({{C3_REPO}}) --"
    [ -d "{{C3_REPO}}" ] && echo "  present" || echo "  missing -> git clone git@github.com:NVIDIA/cosmos.git {{C3_REPO}}"
    echo "-- Framework checkout ({{C3_FRAMEWORK_REPO}}) --"
    [ -d "{{C3_FRAMEWORK_REPO}}" ] && echo "  present" || echo "  missing -> just c3-setup-framework"
    echo "-- Training (SFT) --"
    if [ -d "{{C3_FRAMEWORK_REPO}}/.venv" ]; then
      echo "  framework venv present -> just c3-train-recipes to list SFT recipes"
      echo "  (full SFT tested on 8x H100; convert/config steps run on any GPU)"
    else
      echo "  missing -> just c3-setup-framework"
    fi
    echo "-- Disk free --"
    df -h . | tail -1
    echo "=== done ==="

# Thor/Blackwell ptxas fix (idempotent)
# Triton ships its own `ptxas-blackwell` which on Jetson Thor (compute cap 11.0,
# arch sm_110a) fatals with: "Value 'sm_110a' is not defined for option 'gpu-name'".
# The system CUDA 13 toolkit ptxas DOES support sm_110a. This recipe backs up the
# bundled binary once (.orig) and symlinks it to the system ptxas. Safe to re-run.
c3-fix-ptxas:
    #!/usr/bin/env bash
    set -euo pipefail
    VENV="{{C3_REASON_VENV}}"
    [ -d "$VENV" ] || { echo "venv $VENV missing — run c3-setup-reason first"; exit 0; }
    # Locate triton's bundled ptxas-blackwell inside the venv
    PTXAS="$("$VENV/bin/python" -c "import glob,os;h=glob.glob(os.path.join('$VENV','lib','python*','site-packages','triton','backends','nvidia','bin','ptxas-blackwell'));print(h[0] if h else '')")"
    if [ -z "$PTXAS" ]; then echo "no triton ptxas-blackwell found (nothing to fix)"; exit 0; fi
    # Find a system ptxas that supports sm_110a
    SYS_PTXAS=""
    for cand in /usr/local/cuda-13/bin/ptxas /usr/local/cuda/bin/ptxas "$(command -v ptxas || true)"; do
      [ -x "$cand" ] || continue
      # NOTE: capture to a var (no `grep -q`): grep -q closes the pipe early,
      # sends SIGPIPE to ptxas, and under `set -o pipefail` the pipeline reports
      # failure intermittently (race). Capturing avoids the broken pipe entirely.
      HELP="$("$cand" --help 2>&1 || true)"
      if printf '%s' "$HELP" | grep -q "sm_110a"; then SYS_PTXAS="$cand"; break; fi
    done
    if [ -z "$SYS_PTXAS" ]; then
      echo "⚠ no system ptxas supporting sm_110a found (install CUDA 13 toolkit). Leaving triton ptxas as-is."
      exit 0
    fi
    if [ -L "$PTXAS" ] && [ "$(readlink -f "$PTXAS")" = "$(readlink -f "$SYS_PTXAS")" ]; then
      echo "✅ ptxas already symlinked -> $SYS_PTXAS"; exit 0
    fi
    [ -e "$PTXAS.orig" ] || cp -a "$PTXAS" "$PTXAS.orig"
    ln -sf "$SYS_PTXAS" "$PTXAS"
    echo "✅ patched triton ptxas-blackwell -> $SYS_PTXAS (backup: $PTXAS.orig)"

# Setup: Reasoner (vLLM + vllm-cosmos3)
c3-setup-reason:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
    uv venv --python 3.13 --seed --managed-python "{{C3_REASON_VENV}}"
    source "{{C3_REASON_VENV}}/bin/activate"
    uv pip install --torch-backend={{C3_TORCH_BACKEND}} "vllm=={{C3_VLLM_VERSION}}" \
      "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" \
      openai
    # Thor/Jetson fix: vLLM needs OpenCV to decode video frames server-side
    # (otherwise multimodal video requests fail with "No module named 'cv2'").
    uv pip install opencv-python-headless
    # Thor/Blackwell (sm_110a) fix: Triton's bundled ptxas-blackwell does not
    # recognize sm_110a and aborts PTX codegen. Point it at the system CUDA ptxas.
    just C3_REASON_VENV="{{C3_REASON_VENV}}" c3-fix-ptxas || true
    echo "✅ Reasoner env ready: {{C3_REASON_VENV}}"

# Setup: Generator (Diffusers in-proc)
c3-setup-gen:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
    uv venv --python 3.13 --seed --managed-python "{{C3_GEN_VENV}}"
    source "{{C3_GEN_VENV}}/bin/activate"
    uv pip install --torch-backend={{C3_TORCH_BACKEND}} \
      "diffusers @ git+https://github.com/huggingface/diffusers.git" \
      accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg soundfile \
      torch torchvision transformers
    echo "✅ Generator env ready: {{C3_GEN_VENV}}"

# Setup: vLLM-Omni (Generator server)
c3-setup-omni:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "vLLM-Omni full-modality is easiest via Docker image: vllm/vllm-omni:cosmos3"
    echo "PR-branch (3 merged modes) install:"
    command -v uv >/dev/null || { echo "install uv first"; exit 1; }
    uv venv --python 3.13 --seed --managed-python ".venv-c3-omni"
    source ".venv-c3-omni/bin/activate"
    uv pip install --torch-backend={{C3_TORCH_BACKEND}} \
      "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"
    echo "✅ Omni env ready (text2image/text2video/image2video). For all modalities use the docker image."

# Setup: Cosmos Framework (Action via torchrun)
c3-setup-framework:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{C3_REPO}}/packages"
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || git clone https://github.com/NVIDIA/cosmos-framework.git "{{C3_FRAMEWORK_REPO}}"
    cd "{{C3_FRAMEWORK_REPO}}"
    export GIT_LFS_SKIP_SMUDGE=1
    uv sync --all-extras --group={{C3_TORCH_BACKEND}}-train
    echo "✅ Framework env ready: {{C3_FRAMEWORK_REPO}}/.venv"

# Reasoner: serve (Cosmos3-Nano single GPU)
c3-serve-reason model=C3_MODEL port=C3_REASON_PORT tp="1" max_len="32768" gpu_mem="0.92" enforce_eager="false" offline="false":
    #!/usr/bin/env bash
    set -euo pipefail
    source "{{C3_REASON_VENV}}/bin/activate"
    export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
    # Thor/Blackwell guard: ensure triton ptxas supports sm_110a before launch
    # (no-op on non-Thor; idempotent). Prevents PTXAS "sm_110a not defined" crash.
    just C3_REASON_VENV="{{C3_REASON_VENV}}" c3-fix-ptxas || true
    # Offline mode: skip HF network calls when the model is already cached.
    # Useful on air-gapped/edge boxes (avoids hangs + 401s on gated repos).
    if [ "{{offline}}" = "true" ]; then
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      echo "📴 offline mode: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    fi
    # enforce_eager skips torch.compile/CUDA-graph capture. On some Blackwell/Thor
    # builds the compile path still hits unsupported PTX codegen; eager is a safe
    # fallback (slightly slower, fully functional).
    EAGER_FLAG=""
    [ "{{enforce_eager}}" = "true" ] && EAGER_FLAG="--enforce-eager" && echo "🐢 enforce-eager ON"
    nohup vllm serve "{{model}}" \
      --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
      --tensor-parallel-size {{tp}} \
      --mm-encoder-tp-mode data \
      --async-scheduling \
      --max-model-len {{max_len}} \
      --gpu-memory-utilization {{gpu_mem}} \
      $EAGER_FLAG \
      --allowed-local-media-path / \
      --media-io-kwargs '{"video": {"num_frames": -1}}' \
      --port {{port}} > "{{C3_REASON_LOG}}" 2>&1 &
    echo $! > "{{C3_REASON_PID}}"
    echo "🚀 Reasoner serving (pid $(cat {{C3_REASON_PID}})) on :{{port}} — log: {{C3_REASON_LOG}}"
    echo "   model={{model}}  gpu_mem={{gpu_mem}}  eager={{enforce_eager}}  offline={{offline}}"
    echo "   poll: curl -s localhost:{{port}}/health"
    echo "   Thor tip: if you see PTXAS/sm_110a or OOM, retry with: enforce_eager=true gpu_mem=0.3"

c3-serve-stop-reason:
    #!/usr/bin/env bash
    [ -f "{{C3_REASON_PID}}" ] && kill "$(cat {{C3_REASON_PID}})" 2>/dev/null && rm -f "{{C3_REASON_PID}}" && echo "stopped" || echo "no reason server pid"

c3-serve-status:
    #!/usr/bin/env bash
    # Detect servers by PID file OR by a live HTTP /health on their port.
    # This recognizes both recipe-launched and directly-launched servers.
    check() {
      local name="$1" pf="$2" port="$3"
      if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
        echo "$name: running (pid $(cat "$pf"), port $port)"
      elif curl -sf -o /dev/null "http://localhost:$port/health" 2>/dev/null; then
        echo "$name: running (port $port, no pidfile)"
      else
        echo "$name: stopped"
      fi
    }
    check reason "{{C3_REASON_PID}}" "{{C3_REASON_PORT}}"
    check omni   "{{C3_OMNI_PID}}"   "{{C3_OMNI_PORT}}"

# Reasoner: one-shot inference via our Cosmos3ReasonerModel provider
# untrusted free-text/paths (prompt, image, video, task) are passed via
# environment variables (C3_PROMPT/C3_IMAGE/C3_VIDEO/C3_TASK) and read with
# os.environ inside Python -- NOT via {{param}} interpolation, which was an
# arbitrary-Python-execution sink (CWE-78). Only validated numerics/bools remain
# as positional params.
c3-reason port=C3_REASON_PORT max_tokens="4096" think="false":
    #!/usr/bin/env bash
    set -euo pipefail
    source "{{C3_REASON_VENV}}/bin/activate" 2>/dev/null || true
    C3_THINK="{{think}}" C3_PORT="{{port}}" C3_MAX_TOKENS="{{max_tokens}}" \
      python3 -m strands_cosmos.scripts.c3_cli reason

# Generator: in-proc Diffusers generation
# untrusted free-text/paths (prompt, image, out) are passed via
# environment variables (C3_GEN_PROMPT/C3_GEN_IMAGE/C3_GEN_OUT) and read with
# os.environ inside Python -- NOT via {{param}} interpolation (CWE-78). mode is
# constrained to a fixed set; all other params are validated numerics/bools.
c3-gen mode="text2video" frames="189" fps="24" steps="35" guidance="6.0" res="480" sound="false" seed="0":
    #!/usr/bin/env bash
    set -euo pipefail
    source "{{C3_GEN_VENV}}/bin/activate" 2>/dev/null || true
    C3_GEN_MODE="{{mode}}" C3_GEN_FRAMES="{{frames}}" C3_GEN_FPS="{{fps}}" \
    C3_GEN_STEPS="{{steps}}" C3_GEN_GUIDANCE="{{guidance}}" C3_GEN_RES="{{res}}" \
    C3_GEN_SOUND="{{sound}}" C3_GEN_SEED="{{seed}}" C3_GEN_MODEL="{{C3_MODEL}}" \
      python3 -m strands_cosmos.scripts.c3_cli gen

# Generator server (vLLM-Omni)
c3-serve-omni model=C3_MODEL port=C3_OMNI_PORT:
    #!/usr/bin/env bash
    set -euo pipefail
    source ".venv-c3-omni/bin/activate" 2>/dev/null || true
    nohup vllm serve "{{model}}" --omni \
      --model-class-name Cosmos3OmniDiffusersPipeline \
      --allowed-local-media-path / --port {{port}} --init-timeout 1800 \
      > "{{C3_OMNI_LOG}}" 2>&1 &
    echo $! > "{{C3_OMNI_PID}}"
    echo "🚀 Omni serving (pid $(cat {{C3_OMNI_PID}})) on :{{port}} — log: {{C3_OMNI_LOG}}"

c3-serve-stop-omni:
    #!/usr/bin/env bash
    [ -f "{{C3_OMNI_PID}}" ] && kill "$(cat {{C3_OMNI_PID}})" 2>/dev/null && rm -f "{{C3_OMNI_PID}}" && echo "stopped" || echo "no omni server pid"

# Action / World-Model (Cosmos Framework, torchrun)
# input_jsonl: a JSONL spec, one line per run, with keys:
#   model_mode (forward_dynamics|inverse_dynamics|policy), name, vision_path,
#   action_path (FD/policy), domain_name (av|bridge_orig_lerobot|...),
#   action_chunk_size, fps, image_size, view_point, prompt, seed.
# See cosmos cookbooks/cosmos3/generator/action for sample specs & assets.
c3-action seed="0" preset="latency":
    #!/usr/bin/env bash
    # input_jsonl/out/checkpoint are LLM-controlled paths -> read from
    # env (C3_ACTION_INPUT/OUT/CKPT), never {param}-interpolated (CWE-78).
    # seed (int) + preset (enum) stay positional; preset is validated below.
    set -euo pipefail
    case "{{preset}}" in latency|throughput) ;; *) echo "invalid preset" >&2; exit 2;; esac
    IN="${C3_ACTION_INPUT:?C3_ACTION_INPUT required}"
    OUT="${C3_ACTION_OUT:-/tmp/c3_action}"
    CKPT="${C3_ACTION_CKPT:-Cosmos3-Nano}"
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    .venv/bin/python -m cosmos_framework.scripts.inference \
      --parallelism-preset={{preset}} \
      -i "$IN" \
      -o "$OUT" \
      --checkpoint-path "$CKPT" \
      --seed={{seed}}
    echo "action output -> $OUT  (per-run: <out>/<name>/vision.mp4)"

# Release
# Build sdist+wheel locally. CI (.github/workflows/release.yml) publishes to
# PyPI automatically on a pushed `v*` tag via Trusted Publishing.
build-dist:
    {{PYTHON}} -m pip install --upgrade build
    {{PYTHON}} -m build

# Manual PyPI upload fallback (CI is preferred). Needs ~/.pypirc or TWINE_* env.
publish: build-dist
    {{PYTHON}} -m pip install --upgrade twine
    {{PYTHON}} -m twine upload dist/*


# ══════════════════════════════════════════════════════════════════════════
# Cosmos 3 — Post-Training (Supervised Fine-Tuning) via Cosmos Framework
# Wraps cosmos_framework SFT. Tested upstream on 8× H100 (80GB). On smaller
# GPUs you can validate config/dataset/checkpoint steps; full SFT needs the
# documented multi-GPU allocation.
# ══════════════════════════════════════════════════════════════════════════

C3_TRAIN_NPROC      := env_var_or_default("C3_TRAIN_NPROC", "8")
C3_TRAIN_OUTPUT     := env_var_or_default("C3_TRAIN_OUTPUT", "outputs/train")

# List the available SFT recipes shipped with the framework checkout.
c3-train-recipes:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    echo "=== Cosmos 3 SFT recipes (examples/toml/sft_config) ==="
    ls examples/toml/sft_config/*.toml 2>/dev/null | sed 's#.*/##;s/\.toml$//' || echo "framework not set up -> just c3-setup-framework"
    echo ""
    echo "=== paired launch shells (examples/) ==="
    ls examples/launch_sft_*.sh 2>/dev/null | sed 's#.*/##' || true

# Step 2 — convert a base checkpoint to PyTorch DCP for training.
c3-train-convert checkpoint=C3_MODEL out="":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    name="$(basename '{{checkpoint}}')"
    out="{{out}}"; out="${out:-examples/checkpoints/$name}"
    .venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
      -o "$out" --checkpoint-path "{{checkpoint}}"
    echo "DCP checkpoint -> $out"

# Step 2 (reasoner VLM) — merge Cosmos3 LM onto the Qwen3-VL visual tower.
c3-train-convert-vlm checkpoint=C3_MODEL out="examples/checkpoints/Cosmos3-Nano-VLM":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    .venv/bin/python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
      --checkpoint-path "{{checkpoint}}" -o "{{out}}"
    echo "VLM safetensors -> {{out}}"

# Step 1 helper — turn a captions JSONL into an SFT dataset JSONL.
c3-train-prep-dataset captions out:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    .venv/bin/python -m cosmos_framework.scripts.captions_to_sft_jsonl \
      "$@" 2>/dev/null || .venv/bin/python -m cosmos_framework.scripts.captions_to_sft_jsonl \
      --input "{{captions}}" --output "{{out}}"
    echo "SFT dataset -> {{out}}"

# Validate / export the resolved training config for a recipe TOML (no GPU).
c3-train-show recipe="vision_sft_nano":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    toml="examples/toml/sft_config/{{recipe}}.toml"
    [ -f "$toml" ] || { echo "no such recipe: {{recipe}} (try: just c3-train-recipes)"; exit 1; }
    .venv/bin/python -m cosmos_framework.scripts.train --sft-toml="$toml" --dryrun 2>&1 | tail -60 \
      || { echo "(dryrun unavailable; showing raw recipe TOML)"; cat "$toml"; }

# Step 3 — run SFT. Prefer the paired launch shell (handles paths + checks).
# recipe: vision_sft_nano | vision_sft_super | llava_ov | videophy2_nano
# nproc: GPUs (default 8). dataset/checkpoint: override default paths.
c3-train recipe="vision_sft_nano" nproc=C3_TRAIN_NPROC dataset="" checkpoint="" overrides="":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    launch="examples/launch_sft_{{recipe}}.sh"
    [ -f "$launch" ] || { echo "no launch shell for recipe '{{recipe}}' (try: just c3-train-recipes)"; exit 1; }
    export NPROC_PER_NODE="{{nproc}}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-{{C3_TRAIN_OUTPUT}}}"
    [ -n "{{dataset}}" ]    && export DATASET_PATH="{{dataset}}"
    [ -n "{{checkpoint}}" ] && export BASE_CHECKPOINT_PATH="{{checkpoint}}"
    if [ -n "{{overrides}}" ]; then
      bash "$launch" -- {{overrides}}
    else
      bash "$launch"
    fi

# Step 4 — export a trained DCP checkpoint to HF safetensors.
c3-train-export run_dir out="":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    iter="$(cat '{{run_dir}}/checkpoints/latest_checkpoint.txt')"
    ckpt="{{run_dir}}/checkpoints/$iter"
    out="{{out}}"; out="${out:-{{run_dir}}/hf_export}"
    .venv/bin/python -m cosmos_framework.scripts.export_model \
      --checkpoint-path "$ckpt" -o "$out" 2>/dev/null \
      || .venv/bin/python -m cosmos_framework.scripts.convert_model_to_diffusers \
      --checkpoint-path "$ckpt" -o "$out"
    echo "HF export -> $out"


# Cosmos 3 Generator — vLLM-Omni Docker (full modalities incl. video2video) ──
C3_OMNI_IMAGE  := env_var_or_default("C3_OMNI_IMAGE", "vllm/vllm-omni:cosmos3")
C3_OMNI_WORK   := env_var_or_default("C3_OMNI_WORK", "/tmp/omni-work")

# Start the vLLM-Omni server in Docker (all modalities: t2i/t2v/i2v/v2v/sound/action).
c3-omni-docker model=C3_MODEL port=C3_OMNI_PORT:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{C3_OMNI_WORK}}"
    docker rm -f c3omni 2>/dev/null || true
    docker run -d --name c3omni --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "{{C3_OMNI_WORK}}:/workspace" \
      -p {{port}}:{{port}} --ipc=host \
      "{{C3_OMNI_IMAGE}}" \
      vllm serve "{{model}}" --omni \
      --model-class-name Cosmos3OmniDiffusersPipeline \
      --allowed-local-media-path / --port {{port}} --init-timeout 1800
    echo "🚀 Omni (docker) starting on :{{port}} — poll: curl -s localhost:{{port}}/health"
    echo "   logs: docker logs -f c3omni"

c3-omni-docker-stop:
    -docker rm -f c3omni 2>/dev/null && echo "stopped c3omni" || echo "no c3omni container"

# Video-to-video transfer: re-render an input video with a new prompt (keeps structure).
# Requires the omni server running (c3-omni-docker). input/output are host paths.
c3-v2v input prompt out="/tmp/omni-work/v2v_out.mp4" port=C3_OMNI_PORT steps="35" guidance="8.0" size="832x480" frames="29" fps="16" seed="0" negative="blurry, distorted, low quality" cond_frames="0" cond_keep="last":
    #!/usr/bin/env bash
    # input/prompt/out are untrusted -> passed via env (C3_V2V_*),
    # NOT {param}-interpolated into curl/JSON (CWE-78). Routes through the
    # hardened cosmos3_video2video tool (workspace-confined input+output),
    # the SAME code path the agent uses -- no duplicated curl.
    set -euo pipefail
    C3_V2V_INPUT="{{input}}" C3_V2V_PROMPT="{{prompt}}" C3_V2V_OUT="{{out}}" \
    C3_V2V_PORT="{{port}}" C3_V2V_STEPS="{{steps}}" C3_V2V_GUIDANCE="{{guidance}}" \
    C3_V2V_SIZE="{{size}}" C3_V2V_FRAMES="{{frames}}" C3_V2V_FPS="{{fps}}" \
    C3_V2V_SEED="{{seed}}" C3_V2V_NEGATIVE="{{negative}}" \
    C3_V2V_COND_FRAMES="{{cond_frames}}" C3_V2V_COND_KEEP="{{cond_keep}}" \
      python3 -m strands_cosmos.scripts.c3_cli v2v


# Cosmos 3 — Prompt upsampling / batch captioning / VideoPhy2 eval
# Wrappers over cosmos_framework scripts. The framework venv is set up by
# `just c3-setup-framework`. Reasoner-backed scripts need a reasoner server
# (just c3-serve-reason) on C3_REASON_PORT.

C3_FW_PY := env_var_or_default("C3_FW_PY", C3_FRAMEWORK_REPO + "/.venv/bin/python")

# Prompt upsampling: expand a short scene description into a dense structured
# prompt (Cosmos 3 generator prompt-upsampling). Standalone — builds the
# canonical v4.2 messages and queries the reasoner server (no full sample files).
# task: t2v | t2i | i2v . Needs a reasoner server on `port`.
# untrusted free-text/paths (description, image) are passed via env
# vars (C3_UP_DESC/C3_UP_IMAGE) and read with os.environ -- NOT via {{param}}
# interpolation into the Python heredoc, which was an arbitrary-Python sink
# (CWE-78). task is constrained; numerics/aspect are validated by the caller.
c3-upsample task="t2v" port=C3_REASON_PORT aspect="16,9" width="832" height="480" fps="24" duration="8":
    #!/usr/bin/env bash
    set -euo pipefail
    C3_UP_TASK="{{task}}" C3_UP_PORT="{{port}}" C3_UP_ASPECT="{{aspect}}" \
    C3_UP_WIDTH="{{width}}" C3_UP_HEIGHT="{{height}}" C3_UP_FPS="{{fps}}" \
    C3_UP_DURATION="{{duration}}" "{{C3_FW_PY}}" -m strands_cosmos.scripts.c3_cli upsample

# Batch video captioning via the framework script (reasoner-backed VLM).
# video: a single video OR a directory of videos. Needs a reasoner server.
c3-caption-batch port=C3_REASON_PORT workers="16":
    #!/usr/bin/env bash
    # video/out/template are LLM-controlled paths -> read from env
    # (C3_CAP_VIDEO/OUT/TEMPLATE), never {param}-interpolated (CWE-78).
    set -euo pipefail
    # Resolve the video to an absolute path BEFORE cd (recipe runs from the
    # framework checkout, so a relative video path would not be found there).
    vid="${C3_CAP_VIDEO:?C3_CAP_VIDEO required}"; case "$vid" in /*) ;; *) vid="$(cd "$(dirname "$vid")" && pwd)/$(basename "$vid")";; esac
    out="${C3_CAP_OUT:-/tmp/c3_captions}"; case "$out" in /*) ;; *) out="$(pwd)/$out";; esac
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    # Upstream's default lookup (cosmos_framework/defaults/video_captioner.txt) is
    # wrong in this checkout — the template ships under inference/defaults. Auto-fill
    # it so batch captioning works out of the box; honor an explicit override.
    tmpl="${C3_CAP_TEMPLATE:-}"
    if [ -z "$tmpl" ]; then
      for cand in cosmos_framework/inference/defaults/video_captioner.txt \
                  cosmos_framework/defaults/video_captioner.txt; do
        [ -f "$cand" ] && tmpl="$cand" && break
      done
    fi
    args=(-v "$vid" -o "$out" --server "http://localhost:{{port}}/v1" --max-workers {{workers}})
    [ -n "$tmpl" ] && args+=(--prompt_template_path "$tmpl")
    .venv/bin/python -m cosmos_framework.scripts.caption_from_video "${args[@]}"
    echo "captions -> $out"

# VideoPhy-2 evaluation (Cosmos 3 task-specific eval). Two modes:
#   run+eval:  pass hf_ckpt + val_root (loads HF export, runs, writes summary.json)
#   eval-only: pass only results_dir (re-scores an existing results dir)
c3-eval-videophy2 batch_size="1" max_new_tokens="256" nproc="1":
    #!/usr/bin/env bash
    # results_dir/hf_ckpt/val_root are LLM-controlled paths -> read from
    # env (C3_EVAL_RESULTS/HF_CKPT/VAL_ROOT), never {param}-interpolated (CWE-78).
    set -euo pipefail
    RES="${C3_EVAL_RESULTS:?C3_EVAL_RESULTS required}"
    [ -d "{{C3_FRAMEWORK_REPO}}" ] || { echo "❌ Cosmos Framework missing at {{C3_FRAMEWORK_REPO}} — run: just c3-setup-framework" >&2; exit 1; }
    cd "{{C3_FRAMEWORK_REPO}}"
    args=(--results_dir "$RES" --batch_size {{batch_size}} --max_new_tokens {{max_new_tokens}})
    [ -n "${C3_EVAL_HF_CKPT:-}" ]  && args+=(--hf_ckpt "$C3_EVAL_HF_CKPT")
    [ -n "${C3_EVAL_VAL_ROOT:-}" ] && args+=(--val_root "$C3_EVAL_VAL_ROOT")
    if [ "{{nproc}}" != "1" ]; then
      torchrun --nproc_per_node={{nproc}} -m cosmos_framework.scripts.vlm.eval_videophy2 "${args[@]}"
    else
      .venv/bin/python -m cosmos_framework.scripts.vlm.eval_videophy2 "${args[@]}"
    fi
    echo "videophy2 eval -> $RES/summary.json"

# Generator server (vLLM-Omni) with multi-GPU flags for Cosmos3-Super (64B).
# tp/cfg/ulysses parallelism + optional layerwise offload. Ensure the host has
# at least tp*cfg*ulysses GPUs. Single GPU (Nano): leave all at defaults.
c3-serve-omni-super model="nvidia/Cosmos3-Super" port=C3_OMNI_PORT tp="4" cfg="1" ulysses="1" offload="false":
    #!/usr/bin/env bash
    set -euo pipefail
    source ".venv-c3-omni/bin/activate" 2>/dev/null || true
    extra=""
    [ "{{tp}}" != "1" ]      && extra="$extra --tensor-parallel-size {{tp}}"
    [ "{{cfg}}" != "1" ]     && extra="$extra --cfg-parallel-size {{cfg}}"
    [ "{{ulysses}}" != "1" ] && extra="$extra --ulysses-degree {{ulysses}}"
    [ "{{offload}}" = "true" ] && extra="$extra --enable-layerwise-offload"
    nohup vllm serve "{{model}}" --omni \
      --model-class-name Cosmos3OmniDiffusersPipeline \
      $extra \
      --allowed-local-media-path / --port {{port}} --init-timeout 1800 \
      > "{{C3_OMNI_LOG}}" 2>&1 &
    echo $! > "{{C3_OMNI_PID}}"
    echo "🚀 Omni-Super serving (pid $(cat {{C3_OMNI_PID}})) on :{{port}} [tp={{tp}} cfg={{cfg}} ulysses={{ulysses}} offload={{offload}}]"
