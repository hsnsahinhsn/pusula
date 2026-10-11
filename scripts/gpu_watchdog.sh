#!/usr/bin/env bash
# Keeps a long LLM stage moving on a shared (time-sliced) GPU.
#
# On shared GPUs a long-lived process can lose the device to another tenant and stay blocked for a
# long time, while a fresh process gets it back within seconds. This watchdog watches the stage's
# output files; when they stop growing for STALL seconds it restarts the server and the stage
# (both resumable), then gives them GRACE seconds to start before checking again.
#
#   bash scripts/gpu_watchdog.sh <done-marker> <server-cmd> <stage-cmd> <output-file>...
#   (pass "" as <server-cmd> for stages that need no server, e.g. d3 labeling or training)
#
# Environment: STALL (default 240), GRACE (default 300), CHECK (default 20),
#              MAX_RESTARTS (default 12; then exit 1 so a real bug is not retried forever).
set -uo pipefail

DONE=$1; SERVER_CMD=$2; STAGE_CMD=$3; shift 3
OUTPUTS=("$@")
STALL=${STALL:-240}; GRACE=${GRACE:-300}; CHECK=${CHECK:-20}; MAX_RESTARTS=${MAX_RESTARTS:-12}

log() { echo "[$(date -u +%F' '%T)] $*"; }
lines() { cat "${OUTPUTS[@]}" 2>/dev/null | wc -l; }

restart() {
  log "restart: $1"
  tmux kill-session -t stage 2>/dev/null
  tmux kill-session -t server 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null; pkill -f "python -m pusula" 2>/dev/null; sleep 5
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "python -m pusula" 2>/dev/null; sleep 2
  [[ -n $SERVER_CMD ]] && tmux new-session -d -s server "$SERVER_CMD"
  tmux new-session -d -s stage "$STAGE_CMD"
}

tmux has-session -t stage 2>/dev/null || restart "start"
last=$(lines); since=$(( $(date +%s) + GRACE ))
restarts=0
while [[ ! -f $DONE ]]; do
  sleep "$CHECK"
  now=$(date +%s); cur=$(lines)
  if [[ $cur != "$last" ]]; then
    last=$cur; since=$now
  elif (( now - since > STALL )); then
    restarts=$((restarts + 1))
    if (( restarts > MAX_RESTARTS )); then
      log "giving up after $MAX_RESTARTS restarts at $cur lines"
      tmux kill-session -t stage 2>/dev/null; tmux kill-session -t server 2>/dev/null
      exit 1
    fi
    restart "no progress for $((now - since))s at $cur lines (restart #$restarts)"
    since=$(( $(date +%s) + GRACE ))
  fi
done
log "done marker found: $DONE ($restarts restarts)"
