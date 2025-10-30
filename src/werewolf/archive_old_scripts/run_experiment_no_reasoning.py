#!/usr/bin/env python3
"""
Batch experiment runner for Werewolf game (No Reasoning variant).

Runs multiple games for control and treatment conditions without reasoning strings.
Tests whether Apollo deception probe scores help villagers identify werewolves
when players only provide public statements (no private reasoning).
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_game(config_path: str, game_num: int, condition: str) -> bool:
    """
    Run a single game with the specified config.

    Args:
        config_path: Path to config JSON file
        game_num: Game number for this condition
        condition: 'control' or 'treatment'

    Returns:
        True if game completed successfully, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running {condition.upper()} game {game_num} (NO REASONING)")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    try:
        # Run game_coordinator.py with the specified config
        result = subprocess.run(
            [sys.executable, "game_coordinator.py", config_path],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True
        )

        print(f"✓ {condition.upper()} game {game_num} (NO REASONING) completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ {condition.upper()} game {game_num} (NO REASONING) FAILED")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    """Run the full experiment batch."""

    # Configuration
    NUM_GAMES_PER_CONDITION = 10
    CONTROL_CONFIG = "config.experiment_control_no_reasoning.json"
    TREATMENT_CONFIG = "config.experiment_treatment_no_reasoning.json"

    print(f"""
{'='*70}
WEREWOLF EXPERIMENT BATCH RUNNER (NO REASONING)
{'='*70}

Experimental Design:
  - Control condition: No probe scores shown to players
  - Treatment condition: Probe scores shown with relative scoring explanation
  - NO REASONING: Players only provide public statements (no private reasoning)
  - Games per condition: {NUM_GAMES_PER_CONDITION}
  - Total games: {NUM_GAMES_PER_CONDITION * 2}
  - Game setup: 6 players (2 werewolves, 4 villagers), no seer

Metrics collected:
  - Villager win rate
  - Number of turns to completion
  - Number of werewolves eliminated

Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
""")

    results = {
        'control': {'success': 0, 'failed': 0},
        'treatment': {'success': 0, 'failed': 0}
    }

    # Run control condition games
    print(f"\n\n{'#'*70}")
    print(f"PHASE 1: CONTROL CONDITION (NO REASONING) ({NUM_GAMES_PER_CONDITION} games)")
    print(f"{'#'*70}\n")

    for i in range(1, NUM_GAMES_PER_CONDITION + 1):
        success = run_game(CONTROL_CONFIG, i, 'control')
        if success:
            results['control']['success'] += 1
        else:
            results['control']['failed'] += 1

    # Run treatment condition games
    print(f"\n\n{'#'*70}")
    print(f"PHASE 2: TREATMENT CONDITION (NO REASONING) ({NUM_GAMES_PER_CONDITION} games)")
    print(f"{'#'*70}\n")

    for i in range(1, NUM_GAMES_PER_CONDITION + 1):
        success = run_game(TREATMENT_CONFIG, i, 'treatment')
        if success:
            results['treatment']['success'] += 1
        else:
            results['treatment']['failed'] += 1

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT BATCH COMPLETE (NO REASONING)")
    print(f"{'='*70}\n")
    print(f"Control condition (no reasoning):")
    print(f"  ✓ Successful: {results['control']['success']}/{NUM_GAMES_PER_CONDITION}")
    print(f"  ✗ Failed: {results['control']['failed']}/{NUM_GAMES_PER_CONDITION}")
    print(f"\nTreatment condition (no reasoning):")
    print(f"  ✓ Successful: {results['treatment']['success']}/{NUM_GAMES_PER_CONDITION}")
    print(f"  ✗ Failed: {results['treatment']['failed']}/{NUM_GAMES_PER_CONDITION}")
    print(f"\nTotal successful games: {results['control']['success'] + results['treatment']['success']}/{NUM_GAMES_PER_CONDITION * 2}")
    print(f"\nResults saved to:")
    print(f"  Control: ../../results/experiment_control_no_reasoning/")
    print(f"  Treatment: ../../results/experiment_treatment_no_reasoning/")
    print(f"\nRun analyze_results.py to compute statistics")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
