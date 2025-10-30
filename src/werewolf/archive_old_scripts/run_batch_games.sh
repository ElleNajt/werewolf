#!/bin/bash
# Run 10 games with config.8b.json

for i in {1..10}; do
    echo "=================================="
    echo "Starting game $i of 10"
    echo "=================================="
    ../../venv/bin/python game_coordinator.py config.8b.json
    
    if [ $? -ne 0 ]; then
        echo "Game $i failed, but continuing..."
    else
        echo "Game $i completed successfully"
    fi
    
    echo ""
done

echo "=================================="
echo "All 10 games completed!"
echo "=================================="
