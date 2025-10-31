#!/bin/bash
# Run batch of games where werewolves are told HIGH scores = werewolves (truth)

CONFIG="config.final_8b_probe_truthful_ww.json"
BATCH_NAME="final_8b_probe_truthful_ww"
NUM_GAMES=30

echo "Running $NUM_GAMES games for condition: $BATCH_NAME"
echo "Config: $CONFIG"
echo "Start time: $(date)"

for i in $(seq 0 $((NUM_GAMES - 1))); do
    echo ""
    echo "=== Starting game $i/$((NUM_GAMES - 1)) at $(date) ==="
    ../../venv/bin/python game_coordinator.py "$CONFIG" "$BATCH_NAME"
    
    if [ $? -ne 0 ]; then
        echo "Game $i failed, continuing to next game..."
    fi
done

echo ""
echo "Batch complete: $(date)"
echo "Results directory: ../../results/$BATCH_NAME*"
