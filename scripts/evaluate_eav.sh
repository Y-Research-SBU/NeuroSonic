#!/usr/bin/env bash
set -euo pipefail

EAV_PREPARED_DIR="${EAV_PREPARED_DIR:-/path/to/EAV/prepared_eeg_audio}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/neurosonic_eav}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/neurosonic_eav_eval}"

python train.py \
  --eav_prepared_dir "${EAV_PREPARED_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT_DIR}" \
  --evaluate_gen
