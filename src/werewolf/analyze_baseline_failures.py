#!/usr/bin/env python3
"""
Analyze why 70B baseline has 0% village wins vs 8B's 40%.
Look at voting patterns, eliminations, and game progression.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_game(game_dir: Path):
    """Analyze a single game to understand failure mode."""
    stats_file = game_dir / 'game_stats.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    analysis = {
        'game': game_dir.name,
        'winner': stats['winner'],
        'turns': stats['total_turns'],
        'eliminations': [],
        'vote_distributions': []
    }
    
    # Analyze each turn's elimination
    for player in stats.get('players', []):
        if not player['survived']:
            analysis['eliminations'].append({
                'name': player['name'],
                'role': player['role']
            })
    
    # Get voting patterns from history
    for event in stats.get('history', []):
        if event.get('type') == 'day_elimination':
            data = event.get('data', {})
            votes = data.get('votes', {})
            victim = data.get('victim')
            victim_role = data.get('role')
            
            # Count vote distribution
            vote_counts = defaultdict(int)
            for voted_for in votes.values():
                vote_counts[voted_for] += 1
            
            analysis['vote_distributions'].append({
                'victim': victim,
                'victim_role': victim_role,
                'votes': dict(votes),
                'vote_counts': dict(vote_counts),
                'vote_spread': len(vote_counts)  # How many different players got votes
            })
    
    return analysis


def main():
    print("\n" + "="*80)
    print("ANALYZING 70B BASELINE FAILURE MODE")
    print("="*80)
    
    baseline_70b = Path("../../results/70b_baseline_9622b6b_2025-10-30_17-57-30")
    
    games = sorted([d for d in baseline_70b.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    all_analyses = []
    for game_dir in games:
        analysis = analyze_game(game_dir)
        if analysis:
            all_analyses.append(analysis)
    
    print(f"\nAnalyzed {len(all_analyses)} games")
    print(f"Winners: {sum(1 for a in all_analyses if 'Village' in a['winner'])} village, "
          f"{sum(1 for a in all_analyses if 'Werewolf' in a['winner'])} werewolf")
    
    print("\n" + "="*80)
    print("GAME-BY-GAME BREAKDOWN")
    print("="*80)
    
    for analysis in all_analyses:
        print(f"\n{analysis['game']}: {analysis['winner']} wins in {analysis['turns']} turns")
        elim_str = ", ".join([f"{e['name']} ({e['role']})" for e in analysis['eliminations']])
        print(f"  Eliminations: {elim_str}")
        
        for i, vote_dist in enumerate(analysis['vote_distributions'], 1):
            print(f"  Day {i} voting:")
            print(f"    Eliminated: {vote_dist['victim']} ({vote_dist['victim_role']})")
            print(f"    Vote spread: {vote_dist['vote_spread']} different targets")
            print(f"    Vote counts: {vote_dist['vote_counts']}")
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    total_day_votes = sum(len(a['vote_distributions']) for a in all_analyses)
    villager_eliminations = sum(
        1 for a in all_analyses 
        for v in a['vote_distributions'] 
        if v['victim_role'] == 'villager'
    )
    werewolf_eliminations = sum(
        1 for a in all_analyses 
        for v in a['vote_distributions'] 
        if v['victim_role'] == 'werewolf'
    )
    
    vote_spreads = [
        v['vote_spread'] 
        for a in all_analyses 
        for v in a['vote_distributions']
    ]
    
    print(f"\nTotal day eliminations: {total_day_votes}")
    print(f"  Villagers eliminated: {villager_eliminations} ({100*villager_eliminations/total_day_votes:.1f}%)")
    print(f"  Werewolves eliminated: {werewolf_eliminations} ({100*werewolf_eliminations/total_day_votes:.1f}%)")
    print(f"\nVote spread (# different targets per vote):")
    print(f"  Min: {min(vote_spreads) if vote_spreads else 0}")
    print(f"  Max: {max(vote_spreads) if vote_spreads else 0}")
    print(f"  Average: {sum(vote_spreads)/len(vote_spreads) if vote_spreads else 0:.1f}")
    
    # Look at first day voting specifically
    first_day_votes = [a['vote_distributions'][0] for a in all_analyses if a['vote_distributions']]
    first_day_villager_kills = sum(1 for v in first_day_votes if v['victim_role'] == 'villager')
    
    print(f"\nFirst day eliminations:")
    print(f"  Villagers: {first_day_villager_kills}/{len(first_day_votes)} ({100*first_day_villager_kills/len(first_day_votes):.1f}%)")
    print(f"  Werewolves: {len(first_day_votes) - first_day_villager_kills}/{len(first_day_votes)}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if villager_eliminations == total_day_votes:
        print("\n⚠️  CRITICAL FINDING: Villagers eliminate a villager in 100% of votes!")
        print("    This is not random - there's a systematic coordination failure.")
        print("    Werewolves can vote as a bloc, villagers cannot.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
