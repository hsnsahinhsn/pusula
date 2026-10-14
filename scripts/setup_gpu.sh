#!/usr/bin/env bash
# One-shot setup of a fresh GPU server (CUDA >= 12.8 PyTorch image, e.g. NVIDIA PyTorch NGC) for Pusula
# training and evaluation. Keeps the image's own torch; idempotent.
#
#   bash scripts/setup_gpu.sh            # python deps + Pusula
#   WITH_VLLM=1 bash scripts/setup_gpu.sh   # also vLLM (data generation with Qwen)
#   bash scripts/setup_gpu.sh models     # also download vllm-sr/d3 (to $D3_DIR) and TabiBERT
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} D3_DIR=${D3_DIR:-/workspace/d3}
PIP="python -m pip install -q"

# plain OS images (e.g. Ubuntu 24.04) have no PyTorch: create a venv with a CUDA 13 build (Hopper/Blackwell ready)
if ! python -c "import torch" 2>/dev/null; then
  VENV=${VENV:-$HOME/venv}
  command -v apt-get >/dev/null && { sudo -n true 2>/dev/null && SUDO=sudo || SUDO=;     $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-venv python3-dev build-essential git >/dev/null; }
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install -q --upgrade pip
  python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu130
  echo "source $VENV/bin/activate" >> "${ENV_FILE:-env.sh}"
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
# torch comes from the image (often a pre-release build such as 2.14.0a0+nv): torch-dependent packages are
# installed with --no-deps so pip never tries to replace it; their other dependencies are listed explicitly.
$PIP "transformers==5.17.0" "safetensors>=0.4" einops pyarrow "huggingface_hub>=1.0" pyyaml rich "pydantic>=2"   fastapi uvicorn openai ninja psutil tqdm packaging numpy
$PIP --no-deps "peft>=0.13" "accelerate>=1.10" flash-linear-attention fla-core "bitsandbytes>=0.46.1"  # 4-bit d3 on GPUs < 80 GB
$PIP --no-deps -e .
# the CUDA toolkit that builds kernels must match torch's CUDA (e.g. the image ships 12.6, the torch wheel is 13.0)
TCU=$(python -c "import torch; print(torch.version.cuda)")
if [ ! -x "/usr/local/cuda-$TCU/bin/nvcc" ] && command -v apt-get >/dev/null; then
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-${TCU/./-}" >/dev/null 2>&1 || true
fi
if [ -x "/usr/local/cuda-$TCU/bin/nvcc" ]; then export CUDA_HOME="/usr/local/cuda-$TCU" PATH="/usr/local/cuda-$TCU/bin:$PATH"; fi
# fast causal conv kernel for d3's Gated DeltaNet layers (otherwise a slow fallback is used)
python -c "import causal_conv1d" 2>/dev/null || \
  MAX_JOBS=${MAX_JOBS:-32} $PIP --no-deps --no-build-isolation "causal-conv1d>=1.5"
if [[ ${WITH_VLLM:-0} == 1 ]]; then $PIP vllm; fi  # brings its own torch: use a dedicated image/venv

if [[ ${1:-} == models ]]; then
  hf download vllm-sr/d3 --local-dir "$D3_DIR" --exclude "assets/*"
  hf download boun-tabilab/TabiBERT
fi
python -c "import torch, pusula, transformers, peft, accelerate, fla, causal_conv1d; print('pusula ok, torch', torch.__version__, 'transformers', transformers.__version__)"
