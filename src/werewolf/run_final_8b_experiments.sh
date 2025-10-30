#!/bin/bash
# Final systematic experiments for Llama 8B
# 7 conditions x 10 games each = 70 games total

set -e  # Exit on error

VENV="../../venv/bin/python"
NUM_GAMES=10

echo "=========================================="
echo "FINAL 8B EXPERIMENTS"
echo "7 conditions x ${NUM_GAMES} games = $((7 * NUM_GAMES)) games total"
echo "=========================================="
echo ""

# Condition 1: Baseline (no probe, no CoT)
echo "=========================================="
echo "CONDITION 1: BASELINE"
echo "Private CoT, no probe scores"
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
echo "Private CoT + probe scores (simple directive)"
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
echo "Private CoT + probe scores (cautious framing)"
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

# Condition 4: Public CoT
echo "=========================================="
echo "CONDITION 4: PUBLIC COT"
echo "Public CoT with werewolf confession detection"
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

# Condition 5: Probe simple + CoT monitoring warning
echo "=========================================="
echo "CONDITION 5: PROBE-SIMPLE-MONITORED"
echo "Private CoT + probe + monitoring warning"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running probe-simple-monitored game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_probe_simple_monitored.json final_8b_probe_simple_monitored
    if [ $? -ne 0 ]; then
        echo "ERROR: Probe-simple-monitored game $i failed!"
        exit 1
    fi
done
echo "Probe-simple-monitored complete!"
echo ""

# Condition 6: Probe cautious + CoT monitoring warning
echo "=========================================="
echo "CONDITION 6: PROBE-CAUTIOUS-MONITORED"
echo "Private CoT + probe (cautious) + monitoring warning"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running probe-cautious-monitored game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_probe_cautious_monitored.json final_8b_probe_cautious_monitored
    if [ $? -ne 0 ]; then
        echo "ERROR: Probe-cautious-monitored game $i failed!"
        exit 1
    fi
done
echo "Probe-cautious-monitored complete!"
echo ""

# Condition 7: Public CoT + CoT monitoring warning
echo "=========================================="
echo "CONDITION 7: PUBLIC-COT-MONITORED"
echo "Public CoT + monitoring warning (adversarial)"
echo "=========================================="
for i in $(seq 1 $NUM_GAMES); do
    echo "Running public-cot-monitored game $i/${NUM_GAMES}..."
    $VENV game_coordinator.py config.final_8b_public_cot_monitored.json final_8b_public_cot_monitored
    if [ $? -ne 0 ]; then
        echo "ERROR: Public-cot-monitored game $i failed!"
        exit 1
    fi
done
echo "Public-cot-monitored complete!"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  Baseline:                   results/final_8b_baseline_*/"
echo "  Probe-simple:               results/final_8b_probe_simple_*/"
echo "  Probe-cautious:             results/final_8b_probe_cautious_*/"
echo "  Public-cot:                 results/final_8b_public_cot_*/"
echo "  Probe-simple-monitored:     results/final_8b_probe_simple_monitored_*/"
echo "  Probe-cautious-monitored:   results/final_8b_probe_cautious_monitored_*/"
echo "  Public-cot-monitored:       results/final_8b_public_cot_monitored_*/"
echo ""
echo "To analyze results:"
echo "  for cond in baseline probe_simple probe_cautious public_cot probe_simple_monitored probe_cautious_monitored public_cot_monitored; do"
echo "    echo -n \"\$cond: \""
echo "    grep -r '\"winner\": \"Village\"' results/final_8b_\${cond}_*/ 2>/dev/null | wc -l"
echo "  done"
