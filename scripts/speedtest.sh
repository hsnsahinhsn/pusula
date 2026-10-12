#!/usr/bin/env bash
# Measure training speed on this GPU before a long run: each setting runs a few optimizer steps in its own process
# (an out-of-memory setting simply fails), with the same tokens per optimizer step, and the fastest one is printed.
#
#   bash scripts/speedtest.sh --d3 ~/d3 --precision bf16 --data runs/full/curated --max-prompts 50000
#   # then train with the printed flags added
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/speedtest}; STEPS=${STEPS:-2}
# grad_ckpt:token_budget:accum (8192 x 8 = 65536 tokens per optimizer step everywhere)
CONFIGS=${CONFIGS:-"ckpt:8192:8 ckpt:16384:4 ckpt:32768:2 nockpt:4096:16 nockpt:8192:8 nockpt:16384:4"}
mkdir -p "$OUT"; rm -f "$OUT/bench.jsonl"
for c in $CONFIGS; do
  IFS=: read -r ck tb ac <<< "$c"
  flags="--token-budget $tb --accum $ac"; [[ $ck == nockpt ]] && flags+=" --no-grad-ckpt"
  echo ">> $flags"
  python -m pusula.d3train train "$@" --out "$OUT" --bench "$STEPS" $flags > "$OUT.$ck.$tb.log" 2>&1 \
    || echo "   did not fit / failed (see $OUT.$ck.$tb.log)"
done
python - "$OUT/bench.jsonl" <<'PY'
import json, sys
rows = sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda r: -r["tokens_per_s"])
for r in rows:
    print(f'{"on " if r["grad_ckpt"] else "off"}  budget {r["token_budget"]:>6}  accum {r["accum"]:>2}  '
          f'{r["tokens_per_s"]:>8} tok/s  {r["s_per_step"]:>6} s/step  ~{r["est_hours"]} h  peak {r["peak_gb"]} GB')
b = rows[0]
print("fastest:", f'--token-budget {b["token_budget"]} --accum {b["accum"]}' + ("" if b["grad_ckpt"] else " --no-grad-ckpt"))
PY
