#!/usr/bin/env bash
set -euo pipefail

EAV_PREPARED_DIR="${EAV_PREPARED_DIR:-/path/to/EAV/prepared_eeg_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/neurosonic_eav}"

python train.py \
  --eav_prepared_dir "${EAV_PREPARED_DIR}" \
  --batch_size 32 \
  --epochs 400 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --min_lr 1e-6 \
  --warmup_epochs 0 \
  --online_eval \
  --eval_freq 40 \
  --output_dir "${OUTPUT_DIR}" \
  --auto_split \
  --save_last_freq 50
