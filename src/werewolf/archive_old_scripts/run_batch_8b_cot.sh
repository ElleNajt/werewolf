#!/bin/bash
# Batch experiment: Llama 8B, 2v4, CoT enabled, 10 games

echo "Starting batch experiment: 10 games with Llama 8B (CoT enabled, 2v4)"
echo "Started at: $(date)"

for i in {1..10}; do
    echo ""
    echo "=================================================="
    echo "Running game $i of 10"
    echo "=================================================="
    ../../venv/bin/python game_coordinator.py config.8b.json
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Game $i failed!"
        exit 1
    fi
    
    echo "Game $i completed successfully"
done

echo ""
echo "=================================================="
echo "Batch experiment completed!"
echo "Finished at: $(date)"
echo "=================================================="
