#!/bin/bash
# Batch experiment: Llama 8B, 2v4, CoT only for villagers, probe scores, 10 games

EXPERIMENT_NAME="experiment_8b_villager_cot_only_2v4"
CONFIG_FILE="config.8b_villager_cot_only.json"
NUM_GAMES=10

echo "Starting batch experiment: $NUM_GAMES games with Llama 8B (Villager-only CoT, probe scores, 2v4)"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Started at: $(date)"
echo ""

# Run the games - batch_name parameter creates folder with git hash and timestamp
for i in $(seq 1 $NUM_GAMES); do
    echo "=================================================="
    echo "Running game $i of $NUM_GAMES"
    echo "=================================================="
    ../../venv/bin/python game_coordinator.py $CONFIG_FILE $EXPERIMENT_NAME
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Game $i failed!"
        exit 1
    fi
    
    echo "Game $i completed successfully"
    echo ""
done

echo ""
echo "=================================================="
echo "Batch experiment completed!"
echo "Finished at: $(date)"
echo "Results saved to: ../../results/$EXPERIMENT_NAME*/"
echo "=================================================="
