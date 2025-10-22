#!/usr/bin/env python3
"""
Batch experiment runner for Werewolf game (Outlier Detection variant).

Tests whether telling players to look for OUTLIERS (rather than high scores)
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
    print(f"Running OUTLIER game {game_num}")
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

        print(f"✓ OUTLIER game {game_num} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ OUTLIER game {game_num} FAILED")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    """Run the full experiment batch."""

    # Configuration
    NUM_GAMES = 10
    CONFIG = "config.experiment_outlier.json"

    print(f"""
{'='*70}
WEREWOLF OUTLIER DETECTION EXPERIMENT
{'='*70}

Experimental Design:
  - Probe scores shown BUT computed without role context (ablation)
  - Instructions: "Look for OUTLIERS" instead of "higher = deceptive"
  - Tests: Can outlier framing help when separation is reversed?
  - Games: {NUM_GAMES}
  - Game setup: 6 players (2 werewolves, 4 villagers), no seer

Hypothesis:
  - Ablation shows reversed separation (werewolves score LOWER)
  - Maybe "look for outliers" helps players ignore the direction?
  - Werewolves might still be detectable as outliers even if direction is wrong

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
    print(f"OUTLIER EXPERIMENT BATCH COMPLETE")
    print(f"{'='*70}\n")
    print(f"Outlier condition:")
    print(f"  ✓ Successful: {results['success']}/{NUM_GAMES}")
    print(f"  ✗ Failed: {results['failed']}/{NUM_GAMES}")
    print(f"\nResults saved to: ../../results/experiment_outlier/")
    print(f"\nCompare to:")
    print(f"  Ablation (higher=deceptive): ../../results/experiment_ablation_6player/")
    print(f"  Treatment (with role context): ../../results/experiment_treatment/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
