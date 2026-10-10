#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$ROOT/.server-env/bin/python"
if [[ "${1:-}" == "--worker" ]]; then
    DATA_DIR="$2"
    RUN_DIR="$3"
    export CUDA_VISIBLE_DEVICES="$4"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    exec "$PYTHON" -u "$SCRIPT_DIR/run_detr_comparison.py" \
        --data-root "$DATA_DIR" --manifest "$DATA_DIR/manifest.json" \
        --output-dir "$RUN_DIR" --device cuda:0 --epochs "$5" --init-checkpoint "$6"
fi
PILOT="${1:?Usage: bash run_detr_compare.sh PILOT_DIR GPU_INDEX [EPOCHS] CHECKPOINT}"
GPU_INDEX="${2:-2}"
EPOCHS="${3:-100}"
CHECKPOINT="${4:?Provide pretrained DETR checkpoint path}"
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT"; exit 1; }
CHECKPOINT="$(cd -- "$(dirname -- "$CHECKPOINT")" && pwd)/$(basename -- "$CHECKPOINT")"
[[ "$GPU_INDEX" =~ ^[0-9]+$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 1
DATA_DIR="$(cd -- "$PILOT/data" && pwd)"
[[ -f "$DATA_DIR/manifest.json" && -x "$PYTHON" ]] || exit 1
RUN_DIR="$(mktemp -d "$ROOT/detr-pretrained-XXXXXX")"
nohup bash "$SCRIPT_DIR/run_detr_compare.sh" --worker "$DATA_DIR" "$RUN_DIR" "$GPU_INDEX" "$EPOCHS" "$CHECKPOINT" \
    > "$RUN_DIR/terminal.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/process.pid"
printf 'PID: %s\nRESULT_DIR: %s\nWatch: tail -n 30 -F "%s/terminal.log"\n' "$PID" "$RUN_DIR" "$RUN_DIR"
