#!/usr/bin/env bash
# Pusula: Turkish fine-tuning of Decision 3.0 (vllm-sr/d3), end to end on one GPU server.
# Every stage is resumable and skipped once done.
#
#   bash scripts/pipeline.sh <run-name> [stage ...]
#   stages (default: all, in order):
#     generate   Qwen3.8-27B writes Turkish decision examples (label-first)
#     judge      Qwen re-answers every question blind (second vote)
#     fetch_d3   free the disk from Qwen, download vllm-sr/d3
#     baseline   base d3 (4-bit) on the Turkish Decision Benchmark (the number Pusula must beat)
#     teach      base d3 (4-bit) answers every question (third vote + soft targets)
#     build      consensus labels, train/val/test splits
#     curate     TabiBERT: semantic de-dup, label-noise removal, hard-distractor questions
#     gold       blind gold candidates for an independent annotator (labels go to gold/labels.json)
#     d3train    QLoRA fine-tune of d3 on the curated Turkish data
#     evaluate   base d3 vs Pusula on the test split and the gold benchmark
#
# Environment (optional):
#   N=15000  QWEN_MODEL=Qwen/Qwen3.8-27B-FP8  VLLM_MEM=0.90  D3_DIR=~/d3  HF_HOME=~/hf
#   VLLM_EXTRA="--enforce-eager"   extra vLLM flags (recommended on shared/time-sliced GPUs)
set -euo pipefail

RUN=${1:?usage: pipeline.sh <run-name> [stage ...]}; shift || true
STAGES=${*:-generate judge fetch_d3 baseline teach build curate gold d3train evaluate}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/runs/$RUN"
N=${N:-15000}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}
VLLM_MEM=${VLLM_MEM:-0.90}
VLLM_EXTRA=${VLLM_EXTRA:-}
D3_DIR=${D3_DIR:-$HOME/d3}
export HF_HOME=${HF_HOME:-$HOME/hf}
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false
mkdir -p "$DIR/.done" "$DIR/reports"
cd "$ROOT"

log() { echo "[$(date -u +%F' '%T)] $*"; }
done_mark() { touch "$DIR/.done/$1"; }
is_done() { [[ -f "$DIR/.done/$1" ]]; }

start_vllm() {
  if curl -s -m 3 localhost:8000/v1/models >/dev/null; then return; fi
  log "vLLM başlatılıyor: $QWEN_MODEL"
  # shellcheck disable=SC2086
  nohup vllm serve "$QWEN_MODEL" --served-model-name qwen --port 8000 --max-model-len 16384 \
    --gpu-memory-utilization "$VLLM_MEM" --enable-prefix-caching --max-num-seqs 128 $VLLM_EXTRA \
    >> "$DIR/vllm.log" 2>&1 &
  until curl -s -m 3 localhost:8000/v1/models >/dev/null; do sleep 10; done
}

stop_vllm() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
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
      python -m pusula.judge --inp "$DIR/generated.jsonl" --out "$DIR/judged.jsonl" ;;
    fetch_d3)
      stop_vllm
      rm -rf "$HF_HOME/hub/models--${QWEN_MODEL//\//--}"
      hf download vllm-sr/d3 --local-dir "$D3_DIR" --exclude "assets/*" ;;
    baseline)
      python -m pusula.d3train eval --d3 "$D3_DIR" --data "$ROOT/benchmark/tr_decision_bench_v1.jsonl"         --out "$DIR/reports/d3_base_tr_decision_bench_v1.json" ;;
    teach)
      python -m pusula.teacher --d3 "$D3_DIR" --inp "$DIR/judged.jsonl" --out "$DIR/labeled.jsonl" ;;
    build)
      python -m pusula.dataset --inp "$DIR/labeled.jsonl" --out-dir "$DIR/final" ;;
    curate)
      python -m pusula.curate --data "$DIR/final" --out "$DIR/curated" ;;
    gold)
      python -m pusula.gold export --final "$DIR/curated" --out "$DIR/gold" ;;
    d3train)
      python -m pusula.d3train train --d3 "$D3_DIR" --data "$DIR/curated" --out "$DIR/pusula-d3" ;;
    evaluate)
      for split in "$DIR/curated/test.jsonl" "$DIR/gold/gold.jsonl" "$ROOT/benchmark/tr_decision_bench_v1.jsonl"; do
        [[ -f $split ]] || { log "$split yok, atlandı"; continue; }
        name=$(basename "$split" .jsonl)
        [[ -f "$DIR/reports/d3_base_$name.json" ]] ||           python -m pusula.d3train eval --d3 "$D3_DIR" --data "$split" --out "$DIR/reports/d3_base_$name.json"
        echo "d3_base $name" >> "$DIR/reports/progress.log"
        [[ -f "$DIR/reports/pusula_d3_$name.json" ]] ||           python -m pusula.d3train eval --d3 "$D3_DIR" --adapter "$DIR/pusula-d3/best" --data "$split"             --out "$DIR/reports/pusula_d3_$name.json"
        echo "pusula_d3 $name" >> "$DIR/reports/progress.log"
      done ;;
    *) echo "bilinmeyen aşama: $stage" >&2; exit 2 ;;
  esac
  done_mark "$stage"
done
log "pipeline bitti: $STAGES"
