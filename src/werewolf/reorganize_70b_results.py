#!/usr/bin/env python3
"""
Reorganize 70B experiment results into proper batch directories.
Moves results from results/gameN/ to results/70b_CONDITION_HASH_TIMESTAMP/gameN/
"""

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Game ranges for each condition (based on log analysis)
GAME_RANGES = {
    "70b_baseline": list(range(10, 25)),  # games 10-24 (15 games)
    "70b_public_cot": list(range(25, 40)),  # games 25-39 (15 games)
    "70b_probe_villagers": list(range(40, 55)),  # games 40-54 (15 games)
}

def get_git_hash():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        return "unknown"

def reorganize_70b_results():
    """Reorganize 70B results into batch directories."""
    
    results_dir = Path("../../results")
    git_hash = get_git_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Reorganizing 70B experiment results...")
    print(f"Git hash: {git_hash}")
    print(f"Timestamp: {timestamp}")
    print()
    
    for condition, game_ids in GAME_RANGES.items():
        # Create batch directory
        batch_dir = results_dir / f"{condition}_{git_hash}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {condition}...")
        print(f"  Target directory: {batch_dir}")
        
        moved_count = 0
        for game_id in game_ids:
            source = results_dir / f"game{game_id}"
            target = batch_dir / f"game{game_id - game_ids[0]}"  # Renumber to start from 0
            
            if source.exists():
                print(f"  Moving game{game_id} -> {target.name}")
                shutil.move(str(source), str(target))
                moved_count += 1
            else:
                print(f"  Warning: game{game_id} not found, skipping")
        
        print(f"  Moved {moved_count}/{len(game_ids)} games")
        print()
    
    print("Reorganization complete!")
    print("\nNew directory structure:")
    for condition in GAME_RANGES.keys():
        matching_dirs = list(results_dir.glob(f"{condition}_*"))
        if matching_dirs:
            for d in matching_dirs:
                game_count = len(list(d.glob("game*")))
                print(f"  {d.name}/ ({game_count} games)")

if __name__ == "__main__":
    reorganize_70b_results()
