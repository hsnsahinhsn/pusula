#!/usr/bin/env bash
# One-shot setup of a fresh GPU server (CUDA >= 12.8 PyTorch image, e.g. NVIDIA PyTorch NGC) for Pusula
# training and evaluation. Keeps the image's own torch; idempotent.
#
#   bash scripts/setup_gpu.sh            # python deps + Pusula
#   WITH_VLLM=1 bash scripts/setup_gpu.sh   # also vLLM (data generation with Qwen)
#   bash scripts/setup_gpu.sh models     # also download vllm-sr/d3 (to $D3_DIR), TabiBERT and Qwen3.5-0.8B
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} D3_DIR=${D3_DIR:-/workspace/d3}
PIP="python -m pip install -q"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
# torch comes from the image (often a pre-release build such as 2.14.0a0+nv): torch-dependent packages are
# installed with --no-deps so pip never tries to replace it; their other dependencies are listed explicitly.
$PIP "transformers==5.17.0" "safetensors>=0.4" einops pyarrow "huggingface_hub>=1.0" pyyaml rich "pydantic>=2"   fastapi uvicorn openai ninja psutil tqdm packaging numpy
$PIP --no-deps "peft>=0.13" "accelerate>=1.10" flash-linear-attention fla-core
$PIP --no-deps -e .
# fast causal conv kernel for d3's Gated DeltaNet layers (otherwise a slow fallback is used)
python -c "import causal_conv1d" 2>/dev/null ||   MAX_JOBS=${MAX_JOBS:-32} $PIP --no-deps --no-build-isolation "causal-conv1d>=1.5"
if [[ ${WITH_VLLM:-0} == 1 ]]; then $PIP vllm; fi  # brings its own torch: use a dedicated image/venv

if [[ ${1:-} == models ]]; then
  hf download vllm-sr/d3 --local-dir "$D3_DIR" --exclude "assets/*"
  hf download boun-tabilab/TabiBERT
  hf download Qwen/Qwen3.5-0.8B
fi
python -c "import torch, pusula, transformers, peft, accelerate, fla, causal_conv1d; print('pusula ok, torch', torch.__version__, 'transformers', transformers.__version__)"
