#!/usr/bin/env python3
"""
Batch experiment runner for Werewolf game (Inverted Instructions variant).

Tests whether inverting the instructions ("lower = deceptive" instead of "higher = deceptive")
helps when probe separation is reversed (ablation condition).
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_game(config_path: str, game_num: int) -> bool:
    """
    Run a single game with the specified config.

    Args:
        config_path: Path to config JSON file
        game_num: Game number

    Returns:
        True if game completed successfully, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running INVERTED game {game_num}")
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

        print(f"✓ INVERTED game {game_num} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ INVERTED game {game_num} FAILED")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    """Run the full experiment batch."""

    # Configuration
    NUM_GAMES = 10
    CONFIG = "config.experiment_inverted.json"

    print(f"""
{'='*70}
WEREWOLF INVERTED INSTRUCTIONS EXPERIMENT
{'='*70}

Experimental Design:
  - Probe scores shown BUT computed without role context (ablation)
  - Instructions: "LOWER = deceptive" (inverted from standard)
  - Tests: Can inverting instructions compensate for reversed separation?
  - Games: {NUM_GAMES}
  - Game setup: 6 players (2 werewolves, 4 villagers), no seer

Hypothesis:
  - Ablation shows reversed separation (werewolves score LOWER)
  - Standard "higher = deceptive" instruction fails (0% win rate)
  - Inverted "lower = deceptive" instruction should align with actual signal

Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
""")

    results = {'success': 0, 'failed': 0}

    for i in range(1, NUM_GAMES + 1):
        success = run_game(CONFIG, i)
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"INVERTED EXPERIMENT BATCH COMPLETE")
    print(f"{'='*70}\n")
    print(f"Inverted condition:")
    print(f"  ✓ Successful: {results['success']}/{NUM_GAMES}")
    print(f"  ✗ Failed: {results['failed']}/{NUM_GAMES}")
    print(f"\nResults saved to: ../../results/experiment_inverted/")
    print(f"\nCompare to:")
    print(f"  Ablation (higher=deceptive): ../../results/experiment_ablation_6player/")
    print(f"  Treatment (with role context): ../../results/experiment_treatment/")
    print(f"  Outlier (look for outliers): ../../results/experiment_outlier/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
