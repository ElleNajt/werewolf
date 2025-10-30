#!/bin/bash
# Batch experiment: Llama 8B, 2v4, NO CoT, NO probe scores, 10 games

EXPERIMENT_NAME="experiment_8b_nocot_noprobe_2v4"
CONFIG_FILE="config.8b_nocot_noprobe.json"
NUM_GAMES=10

echo "Starting batch experiment: $NUM_GAMES games with Llama 8B (NO CoT, NO probe scores, 2v4)"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Results will be saved directly to: ../../results/$EXPERIMENT_NAME/"
echo "Started at: $(date)"
echo ""

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

echo "=================================================="
echo "Batch experiment completed!"
echo "Finished at: $(date)"
echo "Results location: ../../results/$EXPERIMENT_NAME/"
echo "=================================================="
