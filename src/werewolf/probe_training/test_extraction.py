#!/usr/bin/env python3
"""
Test script for data extraction pipeline.

This script tests the extraction of training data from werewolf games
without requiring GPU resources.
"""

import sys
from pathlib import Path

from extract_training_data import (
    extract_all_games,
    create_trust_probe_dataset,
    create_collusion_probe_dataset,
    save_dataset_stats
)


def test_extraction(results_dir: Path):
    """Test extracting data from game results."""
    
    print("=" * 80)
    print("TESTING DATA EXTRACTION")
    print("=" * 80)
    
    print(f"\nResults directory: {results_dir}")
    
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        return False
    
    # Extract interactions
    print("\n" + "=" * 80)
    print("STEP 1: Extract all interactions")
    print("=" * 80)
    
    interactions = extract_all_games(results_dir)
    
    if not interactions:
        print("ERROR: No interactions found!")
        return False
    
    print(f"\n✓ Successfully extracted {len(interactions)} interactions")
    
    # Show sample interaction
    print("\nSample interaction:")
    sample = interactions[0]
    print(f"  Game: {sample.game_id}")
    print(f"  Turn: {sample.turn}")
    print(f"  Player: {sample.player} ({sample.player_role})")
    print(f"  Action: {sample.action_type}")
    print(f"  Text: {sample.action[:100]}...")
    if sample.target:
        print(f"  Target: {sample.target} ({sample.target_role})")
        print(f"  Same team: {sample.same_team}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STEP 2: Dataset statistics")
    print("=" * 80)
    
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    save_dataset_stats(interactions, output_dir / "test_stats.json")
    
    # Create trust dataset
    print("\n" + "=" * 80)
    print("STEP 3: Create trust probe dataset")
    print("=" * 80)
    
    trust_dialogues, trust_labels = create_trust_probe_dataset(interactions)
    print(f"\n✓ Created {len(trust_dialogues)} trust probe samples")
    print(f"  Same team (positive): {sum(1 for l in trust_labels if l.value == 'honest')}")
    print(f"  Different team (negative): {sum(1 for l in trust_labels if l.value == 'deceptive')}")
    
    if trust_dialogues:
        print("\nSample trust probe dialogue:")
        sample_dialogue = trust_dialogues[0]
        for msg in sample_dialogue:
            print(f"  {msg.role}: {msg.content[:80]}...")
    
    # Create collusion dataset
    print("\n" + "=" * 80)
    print("STEP 4: Create collusion probe dataset")
    print("=" * 80)
    
    collusion_dialogues, collusion_labels = create_collusion_probe_dataset(interactions)
    print(f"\n✓ Created {len(collusion_dialogues)} collusion probe samples")
    print(f"  Werewolf (positive): {sum(1 for l in collusion_labels if l.value == 'honest')}")
    print(f"  Villager/Seer (negative): {sum(1 for l in collusion_labels if l.value == 'deceptive')}")
    
    if collusion_dialogues:
        print("\nSample collusion probe dialogue:")
        sample_dialogue = collusion_dialogues[0]
        for msg in sample_dialogue:
            print(f"  {msg.role}: {msg.content[:80]}...")
    
    # Check for activations
    print("\n" + "=" * 80)
    print("STEP 5: Check activation availability")
    print("=" * 80)
    
    with_activations = sum(1 for i in interactions if i.activations is not None)
    print(f"\nInteractions with activations: {with_activations}/{len(interactions)}")
    
    if with_activations == 0:
        print("\n⚠ WARNING: No activations found!")
        print("  This data was likely generated without probe backend")
        print("  Training will require re-running games with 'modal_probe' backend")
    else:
        print(f"\n✓ {with_activations} interactions have activation data")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data extraction pipeline")
    parser.add_argument("--results-dir", type=str, 
                        default="../../results/experiment_treatment",
                        help="Directory containing game results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    success = test_extraction(results_dir)
    
    sys.exit(0 if success else 1)
