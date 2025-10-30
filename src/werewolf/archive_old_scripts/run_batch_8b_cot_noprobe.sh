#!/bin/bash
# Batch experiment: Llama 8B, 2v4, CoT enabled, NO probe scores, 10 games

EXPERIMENT_NAME="experiment_8b_cot_noprobe_2v4"
CONFIG_FILE="config.8b_cot_noprobe.json"
NUM_GAMES=10

echo "Starting batch experiment: $NUM_GAMES games with Llama 8B (CoT enabled, NO probe scores, 2v4)"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Started at: $(date)"

# Get the starting game number
LAST_GAME=$(ls -d ../../results/game* 2>/dev/null | sed 's/.*game//' | sort -n | tail -1)
if [ -z "$LAST_GAME" ]; then
    START_GAME=1
else
    START_GAME=$((LAST_GAME + 1))
fi
END_GAME=$((START_GAME + NUM_GAMES - 1))

echo "Games will be numbered: game${START_GAME} to game${END_GAME}"
echo ""

# Run the games
for i in $(seq 1 $NUM_GAMES); do
    echo "=================================================="
    echo "Running game $i of $NUM_GAMES"
    echo "=================================================="
    ../../venv/bin/python game_coordinator.py $CONFIG_FILE
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Game $i failed!"
        exit 1
    fi
    
    echo "Game $i completed successfully"
    echo ""
done

# Move games to experiment folder
echo "=================================================="
echo "Organizing results into experiment folder..."
echo "=================================================="

mkdir -p ../../results/$EXPERIMENT_NAME

for game_num in $(seq $START_GAME $END_GAME); do
    if [ -d "../../results/game${game_num}" ]; then
        mv ../../results/game${game_num} ../../results/$EXPERIMENT_NAME/
        echo "Moved game${game_num} to $EXPERIMENT_NAME/"
    fi
done

echo ""
echo "=================================================="
echo "Batch experiment completed!"
echo "Finished at: $(date)"
echo "Results saved to: ../../results/$EXPERIMENT_NAME/"
echo "=================================================="
