#!/usr/bin/env python3
"""
Simple test script for data extraction that doesn't require heavy dependencies.

This script tests the extraction logic by directly processing game files.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_game_stats(game_dir: Path):
    """Load game statistics."""
    stats_file = game_dir / "game_stats.json"
    
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def extract_interactions_simple(stats):
    """Extract interactions from game stats without deception-detection dependencies."""
    interactions = []
    
    # Build role mapping
    roles = {}
    for player in stats['players']:
        roles[player['name']] = player['role']
    
    # Extract from history
    for event in stats['history']:
        event_type = event['type']
        data = event['data']
        turn = event['turn']
        
        if event_type == 'day_statement':
            player = data['player']
            statement = data['statement']
            interactions.append({
                'type': 'statement',
                'player': player,
                'role': roles[player],
                'turn': turn,
                'text': statement
            })
        
        elif event_type == 'day_elimination':
            # Extract votes
            votes = data.get('votes', {})
            for voted_for, count in votes.items():
                if voted_for in roles:
                    # Record that someone voted for this player
                    interactions.append({
                        'type': 'vote',
                        'target': voted_for,
                        'target_role': roles[voted_for],
                        'turn': turn,
                        'vote_count': count
                    })
        
        elif event_type == 'night_kill':
            werewolf_votes = data.get('werewolf_votes', {})
            for werewolf, target in werewolf_votes.items():
                interactions.append({
                    'type': 'night_kill',
                    'player': werewolf,
                    'role': roles[werewolf],
                    'target': target,
                    'target_role': roles[target],
                    'turn': turn
                })
    
    return interactions


def analyze_results_dir(results_dir: Path):
    """Analyze a results directory."""
    game_dirs = sorted([d for d in results_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('game')])
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {results_dir}")
    print(f"{'='*80}\n")
    
    if not game_dirs:
        print("No game directories found!")
        return
    
    print(f"Found {len(game_dirs)} games\n")
    
    total_interactions = 0
    by_type = defaultdict(int)
    by_role = defaultdict(int)
    games_with_activations = 0
    total_activations = 0
    
    for game_dir in game_dirs:
        stats = load_game_stats(game_dir)
        if not stats:
            print(f"  {game_dir.name}: No stats file")
            continue
        
        interactions = extract_interactions_simple(stats)
        total_interactions += len(interactions)
        
        for interaction in interactions:
            by_type[interaction['type']] += 1
            if 'role' in interaction:
                by_role[interaction['role']] += 1
        
        # Check for activations
        player_acts = stats.get('player_activations', {})
        if player_acts:
            games_with_activations += 1
            for player, acts in player_acts.items():
                total_activations += len(acts)
        
        print(f"  {game_dir.name}: {len(interactions)} interactions, "
              f"{sum(len(acts) for acts in player_acts.values())} activations")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total interactions: {total_interactions}")
    print(f"\nBy type:")
    for itype, count in sorted(by_type.items()):
        print(f"  {itype}: {count}")
    
    print(f"\nBy role:")
    for role, count in sorted(by_role.items()):
        print(f"  {role}: {count}")
    
    print(f"\nActivations:")
    print(f"  Games with activations: {games_with_activations}/{len(game_dirs)}")
    print(f"  Total activations: {total_activations}")
    
    if total_activations == 0:
        print(f"\n⚠️  WARNING: No activation data found!")
        print(f"  These games were likely run without probe backend enabled.")
        print(f"  To train probes, you need games with 'modal_probe' backend.")
    else:
        print(f"\n✅ Activation data available for probe training!")
    
    # Show sample data structure
    if game_dirs:
        first_game = game_dirs[0]
        stats = load_game_stats(first_game)
        if stats:
            print(f"\n{'='*80}")
            print("SAMPLE GAME STRUCTURE")
            print(f"{'='*80}\n")
            print(f"Game ID: {stats.get('game_id')}")
            print(f"Winner: {stats.get('winner')}")
            print(f"Total turns: {stats.get('total_turns')}")
            print(f"Players: {len(stats.get('players', []))}")
            
            if stats.get('player_activations'):
                print(f"\nSample activation:")
                for player, acts in stats['player_activations'].items():
                    if acts:
                        sample_act = acts[0]
                        print(f"  Player: {player}")
                        print(f"  Action: {sample_act.get('action', 'N/A')[:60]}...")
                        if 'activations' in sample_act:
                            act_data = sample_act['activations']
                            print(f"  Score: {act_data.get('aggregate_score', 'N/A')}")
                            print(f"  Tokens: {act_data.get('num_tokens', 'N/A')}")
                        break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze werewolf game data")
    parser.add_argument("--results-dir", type=str,
                        default="/workspace/results/experiment_treatment",
                        help="Directory containing game results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        exit(1)
    
    analyze_results_dir(results_dir)
