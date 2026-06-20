#!/usr/bin/env bash
# Fast training run for quick iteration.
# Preprocesses 500 matches (~4% of the dataset) and trains for up to 5 epochs
# with early stopping. Should finish in under 10 minutes.

set -e
source .venv/bin/activate

PROCESSED_DIR="./data/processed_fast"
CHECKPOINT_DIR="./checkpoints_fast"
LOG_DIR="./runs_fast"

mkdir -p "$PROCESSED_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

echo "=== Preprocessing 500 matches ==="
python preprocess.py \
  --data_dir /home/jack/workspace/open-data/data \
  --out_dir "$PROCESSED_DIR" \
  --max_matches 500 \
  --num_workers 8

echo "=== Training ==="
python train.py \
  --processed_dir "$PROCESSED_DIR" \
  --epochs 5 \
  --batch_size 128 \
  --seq_len 50 \
  --seq_stride 25 \
  --horizon 5.0 \
  --val_ratio 0.15 \
  --num_workers 8 \
  --lr 1e-3 \
  --patience 2 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --log_dir "$LOG_DIR"
