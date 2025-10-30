#!/bin/bash
# Script to run trust probe training on Modal
# Run from: /workspace/src/werewolf/probe_training
#
# First time setup:
#   ../../../probe_venv/bin/modal token new
#
# Then run this script to train the probe

cd "$(dirname "$0")"  # Go to script directory

../../../probe_venv/bin/modal run train_on_modal.py train \
  --results-dir ../../../results/experiment_treatment \
  --probe-type trust \
  --layers 22 \
  --batch-size 4

echo ""
echo "Training complete! Probe saved to ./probes/trust_lr_l22/detector.pt"
