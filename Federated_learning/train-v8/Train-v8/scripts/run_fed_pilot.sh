#!/usr/bin/env bash
# One exploratory DETR FedAvg round: name-proxy split, random model initialization.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd -- "$CODE_DIR/.." && pwd)"
PYTHON="$ROOT/.server-env/bin/python"

if [[ "${1:-}" == "--worker" ]]; then
    RUN_DIR="$2"
    export CUDA_VISIBLE_DEVICES="$3"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    cd "$ROOT"
    echo 'Exploratory pilot: proxy groups, new split, random initialization, one round.'
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
    "$PYTHON" -u "$SCRIPT_DIR/prepare_federated_data.py" \
        --source "$ROOT/Data/detr_data" --output-dir "$RUN_DIR/data" --allow-name-proxy
    "$PYTHON" -u "$SCRIPT_DIR/run_federated.py" \
        --arch detr --manifest "$RUN_DIR/data/manifest.json" --data-root "$RUN_DIR/data" \
        --output-dir "$RUN_DIR/model" --config "$CODE_DIR/configs/fl_pilot.yaml" \
        --method fedavg --rounds 1 --local-epochs 1 --num-clients 3 \
        --covariate gap_days --device cuda:0 --num-workers 0 --exploratory
    echo 'COMPLETED PILOT'
    exit 0
fi

GPU_INDEX="${1:-2}"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'Usage: bash run_fed_pilot.sh GPU_INDEX'; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON"; exit 1; }
[[ -f "$ROOT/Data/detr_data/annotations/train.csv" ]] || { echo 'Missing DETR data'; exit 1; }
RUN_DIR="$(mktemp -d "$ROOT/fedavg-pilot-XXXXXX")"
nohup bash "$SCRIPT_DIR/run_fed_pilot.sh" --worker "$RUN_DIR" "$GPU_INDEX" \
    > "$RUN_DIR/terminal.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/process.pid"
printf 'PID: %s\nRESULT_DIR: %s\nWatch: tail -n 30 -F "%s/terminal.log"\n' "$PID" "$RUN_DIR" "$RUN_DIR"
