#!/usr/bin/env python3
"""
Test the backend integration by running a short game with Claude backend.
"""

from config import GameConfig
from game_coordinator import GameCoordinator

if __name__ == "__main__":
    print("Testing backend abstraction with Claude backend...")

    # Create config for Claude backend
    config = GameConfig(
        backend_type="claude",
        output_dir="/workspace/werewolf_game",
        max_turns=2  # Just 2 turns for quick test
    )

    # Run game
    coordinator = GameCoordinator(config=config)
    result = coordinator.run_game()

    print(f"\nTest complete! Result: {result}")
    print(f"Game saved to: {coordinator.output_dir}")
