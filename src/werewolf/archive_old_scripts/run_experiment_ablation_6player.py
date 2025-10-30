#!/usr/bin/env python3
"""
Batch experiment runner for Werewolf game (Ablation - no role context in probe scoring).

Runs treatment condition games where probe scores are computed WITHOUT role context.
This tests whether the probe detects actual deception or just "lying instructions in prompt".
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
    print(f"Running ABLATION game {game_num} (6 players, no role context in scoring)")
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

        print(f"✓ ABLATION game {game_num} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ ABLATION game {game_num} FAILED")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    """Run the full experiment batch."""

    # Configuration
    NUM_GAMES = 10
    CONFIG = "config.experiment_ablation_6player.json"

    print(f"""
{'='*70}
WEREWOLF ABLATION EXPERIMENT BATCH RUNNER (6 players)
{'='*70}

Experimental Design:
  - Ablation: Probe scores shown BUT computed without role context
  - This tests: Does probe detect deception or just "lying instructions"?
  - Games: {NUM_GAMES}
  - Game setup: 6 players (2 werewolves, 4 villagers), no seer

Key difference from treatment:
  - Generation: WITH role context ("You are a WEREWOLF")
  - Scoring: WITHOUT role context (role info stripped before probe)

If probe still works → detecting actual deceptive behavior
If probe fails → just detecting "lying instructions in prompt"

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
    print(f"ABLATION EXPERIMENT BATCH COMPLETE (6 players)")
    print(f"{'='*70}\n")
    print(f"Ablation condition:")
    print(f"  ✓ Successful: {results['success']}/{NUM_GAMES}")
    print(f"  ✗ Failed: {results['failed']}/{NUM_GAMES}")
    print(f"\nResults saved to: ../../results/experiment_ablation_6player/")
    print(f"\nCompare to treatment (with role context): ../../results/experiment_treatment/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
