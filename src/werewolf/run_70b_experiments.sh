#!/bin/bash
# Run all three 70B experimental conditions (15 games each)

echo "=========================================="
echo "70B WEREWOLF EXPERIMENTS"
echo "=========================================="
echo "Starting at: $(date)"
echo ""
echo "This will run 3 conditions × 15 games = 45 total games"
echo ""

# Condition 1: Baseline (no help)
echo "=========================================="
echo "CONDITION 1: 70B Baseline (CoT private, no probe)"
echo "=========================================="
./run_batch_70b_baseline.sh
echo ""

# Condition 2: Public CoT
echo "=========================================="
echo "CONDITION 2: 70B Public CoT (reasoning visible)"
echo "=========================================="
./run_batch_70b_public_cot.sh
echo ""

# Condition 3: Probe with villager instructions
echo "=========================================="
echo "CONDITION 3: 70B Probe + Villager Instructions"
echo "=========================================="
./run_batch_70b_probe_villagers.sh
echo ""

echo "=========================================="
echo "ALL 70B EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Finished at: $(date)"
echo "Total games: 45 (3 conditions × 15 games)"
