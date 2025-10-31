#!/bin/bash
# Run 15 games for 70B probe with villager instructions condition

CONFIG="config.70b_probe_villagers.json"
NUM_GAMES=15
CONDITION="70b_probe_villagers"

echo "Running $NUM_GAMES games for condition: $CONDITION"
echo "Config: $CONFIG"
echo "Start time: $(date)"

for i in $(seq 0 $((NUM_GAMES - 1))); do
    echo ""
    echo "=== Starting game $i/$((NUM_GAMES - 1)) at $(date) ==="
    ../../venv/bin/python game_coordinator.py "$CONFIG"
    echo "=== Finished game $i/$((NUM_GAMES - 1)) at $(date) ==="
done

echo ""
echo "All $NUM_GAMES games completed at $(date)"
