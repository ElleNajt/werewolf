#!/bin/bash
# Final systematic experiments for Llama 8B
# 4 conditions x 10 games each = 40 games total

set -e  # Exit on error

VENV="../../venv/bin/python"
NUM_GAMES=10

echo "=========================================="
echo "FINAL 8B EXPERIMENTS"
echo "4 conditions x ${NUM_GAMES} games = $((4 * NUM_GAMES)) games total"
echo "=========================================="
echo ""

# Condition 1: Baseline (no probe, no CoT)
echo "=========================================="
echo "CONDITION 1: BASELINE"
echo "No probe scores, no CoT"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running baseline game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_baseline.json final_8b_baseline
    if [ $? -ne 0 ]; then
        echo "ERROR: Baseline game $i failed!"
        exit 1
    fi
done
echo "Baseline complete!"
echo ""

# Condition 2: Probe with simple directive
echo "=========================================="
echo "CONDITION 2: PROBE (SIMPLE)"
echo "Probe scores with simple directive"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running probe-simple game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_probe_simple.json final_8b_probe_simple
    if [ $? -ne 0 ]; then
        echo "ERROR: Probe-simple game $i failed!"
        exit 1
    fi
done
echo "Probe-simple complete!"
echo ""

# Condition 3: Probe with cautious framing
echo "=========================================="
echo "CONDITION 3: PROBE (CAUTIOUS)"
echo "Probe scores with incorporate/cautious framing"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running probe-cautious game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_probe_cautious.json final_8b_probe_cautious
    if [ $? -ne 0 ]; then
        echo "ERROR: Probe-cautious game $i failed!"
        exit 1
    fi
done
echo "Probe-cautious complete!"
echo ""

# Condition 4: Public CoT with werewolf confession instruction
echo "=========================================="
echo "CONDITION 4: PUBLIC COT"
echo "Public CoT with confession detection instruction"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running public-cot game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_public_cot.json final_8b_public_cot
    if [ $? -ne 0 ]; then
        echo "ERROR: Public-cot game $i failed!"
        exit 1
    fi
done
echo "Public-cot complete!"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  Baseline:       results/final_8b_baseline_*/"
echo "  Probe-simple:   results/final_8b_probe_simple_*/"
echo "  Probe-cautious: results/final_8b_probe_cautious_*/"
echo "  Public-cot:     results/final_8b_public_cot_*/"
echo ""
echo "To analyze results, run:"
echo "  grep -r '\"winner\"' results/final_8b_*/ | grep -c Village"
echo "  grep -r '\"winner\"' results/final_8b_*/ | grep -c Werewolves"
