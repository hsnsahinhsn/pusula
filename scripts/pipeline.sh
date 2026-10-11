#!/usr/bin/env bash
# Pusula training pipeline on a single GPU server. Every stage is resumable and skipped once done.
#
#   bash scripts/pipeline.sh <run-name> [stage ...]
#   stages: generate judge teach build gold train evaluate   (default: all, in order)
#
# Environment (optional):
#   N=30000                 examples to generate
#   QWEN_MODEL=Qwen/Qwen3.8-27B-FP8   generator/judge model served by vLLM
#   VLLM_MEM=0.90           vLLM --gpu-memory-utilization
#   HF_HOME=~/hf            Hugging Face cache (models are deleted from it between stages)
set -euo pipefail

RUN=${1:?usage: pipeline.sh <run-name> [stage ...]}; shift || true
STAGES=${*:-generate judge teach build gold train evaluate}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/runs/$RUN"
N=${N:-30000}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}
VLLM_MEM=${VLLM_MEM:-0.90}
export HF_HOME=${HF_HOME:-$HOME/hf}
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false
mkdir -p "$DIR/.done"
cd "$ROOT"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
done_mark() { touch "$DIR/.done/$1"; }
is_done() { [[ -f "$DIR/.done/$1" ]]; }
hf_delete() { rm -rf "$HF_HOME/hub/models--${1//\//--}"; }

start_vllm() {
  if curl -s -m 3 localhost:8000/v1/models >/dev/null; then return; fi
  log "vLLM başlatılıyor: $QWEN_MODEL"
  nohup vllm serve "$QWEN_MODEL" --served-model-name qwen --port 8000 --max-model-len 16384 \
    --gpu-memory-utilization "$VLLM_MEM" --enable-prefix-caching --max-num-seqs 128 \
    > "$DIR/vllm.log" 2>&1 &
  echo $! > "$DIR/vllm.pid"
  until curl -s -m 3 localhost:8000/v1/models >/dev/null; do sleep 10; done
}

stop_vllm() {
  [[ -f "$DIR/vllm.pid" ]] && kill "$(cat "$DIR/vllm.pid")" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
}

for stage in $STAGES; do
  if is_done "$stage"; then log "$stage: tamam, atlanıyor"; continue; fi
  log "== $stage =="
  case $stage in
    generate)
      start_vllm
      python -m pusula.generate --n "$N" --out "$DIR/generated.jsonl" ;;
    judge)
      start_vllm
      python -m pusula.judge --inp "$DIR/generated.jsonl" --out "$DIR/judged.jsonl"
      stop_vllm; hf_delete "$QWEN_MODEL" ;;
    teach)
      stop_vllm
      python -m pusula.teacher --inp "$DIR/judged.jsonl" --out "$DIR/labeled.jsonl" \
        --raw-log "$DIR/d3_first_response.json"
      hf_delete vllm-sr/d3 ;;
    build)
      python -m pusula.dataset --inp "$DIR/labeled.jsonl" --out-dir "$DIR/final" ;;
    gold)
      python -m pusula.gold export --final "$DIR/final" --out "$DIR/gold/candidates.json" ;;
    train)
      python -m pusula.train --data "$DIR/final" --out "$DIR/model" ;;
    evaluate)
      python -m pusula.evaluate --model "$DIR/model/best" --out "$DIR/reports" \
        --data "$DIR/final/test.jsonl" "$DIR/gold/gold.jsonl" ;;
    *) echo "bilinmeyen aşama: $stage" >&2; exit 2 ;;
  esac
  done_mark "$stage"
done
log "pipeline bitti: $STAGES"
