#!/usr/bin/env bash
set -euo pipefail

# MLX LoRA smoke training script (Apple Silicon).
#
# Requires:
# - `mlx_lm` installed (pip)
# - a dataset directory produced by `docagent make-trainset-dir`
#
# Example:
#   ./scripts/smoke.sh ./data/smoke/docs.db
#   docagent make-trainset-dir --db ./data/smoke/docs.db --out-dir ./data/trainset --n 2000
#   ./scripts/mlx_lora_smoke.sh ./data/trainset ./data/adapters/docagent-qwen0_5b

DATA_DIR="${1:-./data/trainset}"
ADAPTER_PATH="${2:-./data/adapters/docagent-qwen0_5b}"

mkdir -p "$(dirname "$ADAPTER_PATH")"

mlx_lm.lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data "$DATA_DIR" \
  --iters 50 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --steps-per-report 5 \
  --steps-per-eval 25 \
  --val-batches 1 \
  --max-seq-length 1024 \
  --num-layers 4 \
  --mask-prompt \
  --grad-checkpoint \
  --adapter-path "$ADAPTER_PATH"

echo "OK: adapters saved to $ADAPTER_PATH"

