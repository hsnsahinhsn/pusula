#!/usr/bin/env bash
# Runs pipeline stages one after another; GPU stages run under scripts/gpu_watchdog.sh so a stage that
# stops making progress on a shared GPU is restarted (every stage is resumable).
#
#   bash scripts/pipeline_watched.sh <run-name> [stage ...]
#
# Optional: WAIT_FOR=<file>  wait until this file exists before starting (e.g. another runner's marker).
set -uo pipefail

RUN=${1:?usage: pipeline_watched.sh <run-name> [stage ...]}; shift || true
STAGES=${*:-generate judge fetch_d3 baseline teach build curate gold d3train evaluate}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/runs/$RUN"
mkdir -p "$DIR/.done"
cd "$ROOT"

log() { echo "[$(date -u +%F' '%T)] $*"; }

if [[ -n ${WAIT_FOR:-} ]]; then
  log "bekleniyor: $WAIT_FOR"
  until [[ -f $WAIT_FOR ]]; do sleep 60; done
fi

# stage -> files whose line count shows progress, stall seconds, grace seconds (empty = no watchdog)
watch_spec() {
  case $1 in
    baseline) echo "$DIR/reports/d3_base_tr_decision_bench_v1.json|3600|1800" ;;
    teach)    echo "$DIR/labeled.jsonl|900|1200" ;;
    curate)   echo "$DIR/curated/noise/fold0/model/train_log.jsonl $DIR/curated/noise/fold1/model/train_log.jsonl $DIR/curated/noise/fold2/model/train_log.jsonl $DIR/curated/train.jsonl|1800|1200" ;;
    d3train)  echo "$DIR/pusula-d3/train_log.jsonl|1800|1800" ;;
    evaluate) echo "$DIR/reports/progress.log|3600|1800" ;;
    *)        echo "" ;;
  esac
}

for stage in $STAGES; do
  if [[ -f "$DIR/.done/$stage" ]]; then log "$stage: tamam, atlanıyor"; continue; fi
  spec=$(watch_spec "$stage")
  log "== $stage =="
  if [[ -z $spec ]]; then
    bash scripts/pipeline.sh "$RUN" "$stage" || { log "$stage başarısız"; exit 1; }
  else
    IFS='|' read -r outputs stall grace <<< "$spec"
    # shellcheck disable=SC2086
    STALL=$stall GRACE=$grace bash scripts/gpu_watchdog.sh "$DIR/.done/$stage" "" \
      "bash scripts/pipeline.sh $RUN $stage >> $DIR/$stage.log 2>&1" $outputs
  fi
  [[ -f "$DIR/.done/$stage" ]] || { log "$stage tamamlanmadı"; exit 1; }
done
log "zincir bitti: $STAGES"
