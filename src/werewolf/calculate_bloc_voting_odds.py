#!/usr/bin/env python3
"""
Calculate the probability of eliminating a villager when:
- 2 werewolves vote as a coordinated bloc (both vote for same target)
- 4 villagers vote randomly/independently
"""

import itertools
from collections import Counter


def simulate_voting_outcomes():
    """
    Simulate all possible voting outcomes when:
    - 2 werewolves coordinate (vote for same player)
    - 4 villagers vote independently
    
    Returns probability of villager elimination.
    """
    
    # Players: 4 villagers, 2 werewolves
    villagers = ['V1', 'V2', 'V3', 'V4']
    werewolves = ['W1', 'W2']
    all_players = villagers + werewolves
    
    # Werewolves can target any of the 4 villagers
    # (They won't target each other in optimal play)
    werewolf_targets = villagers
    
    villager_elimination_count = 0
    total_outcomes = 0
    
    outcome_details = []
    
    # For each possible werewolf target
    for ww_target in werewolf_targets:
        # For each possible combination of villager votes
        # Each villager can vote for any of the other 5 players
        # (In practice, voting for themselves is rare, but let's include all possibilities)
        
        # Each of 4 villagers picks from 5 targets (all others)
        for v1_vote in [p for p in all_players if p != 'V1']:
            for v2_vote in [p for p in all_players if p != 'V2']:
                for v3_vote in [p for p in all_players if p != 'V3']:
                    for v4_vote in [p for p in all_players if p != 'V4']:
                        
                        # Count votes
                        votes = Counter()
                        votes[ww_target] += 2  # Both werewolves vote together
                        votes[v1_vote] += 1
                        votes[v2_vote] += 1
                        votes[v3_vote] += 1
                        votes[v4_vote] += 1
                        
                        # Find who gets eliminated (most votes)
                        max_votes = max(votes.values())
                        candidates_for_elimination = [p for p, v in votes.items() if v == max_votes]
                        
                        # If there's a tie, pick one (in practice, game would have tie-breaking)
                        # For this analysis, we'll count each tied candidate proportionally
                        for candidate in candidates_for_elimination:
                            outcome_weight = 1.0 / len(candidates_for_elimination)
                            
                            if candidate in villagers:
                                villager_elimination_count += outcome_weight
                            
                            total_outcomes += outcome_weight
    
    prob_villager_elimination = villager_elimination_count / total_outcomes
    return prob_villager_elimination, total_outcomes


def calculate_simplified():
    """
    Simplified calculation assuming villagers vote uniformly at random.
    """
    print("\n" + "="*80)
    print("SIMPLIFIED ANALYTICAL CALCULATION")
    print("="*80)
    
    print("\nSetup:")
    print("- 2 werewolves vote together for same villager (2 votes)")
    print("- 4 villagers each vote independently for random other player")
    print("- 6 total players voting, 6 possible targets for each voter")
    
    # Werewolves pick a villager and both vote for them: 2 votes
    # Each villager votes for one of 5 other players uniformly at random
    
    # Probability that werewolves' target gets eliminated:
    # This happens if no other player gets more than 2 votes
    # Or if there's a tie at 2+ votes and the werewolf target is selected
    
    # Let's think about it differently:
    # - Werewolf target starts with 2 votes
    # - Each of 4 villagers distributes 1 vote among 5 players
    # - Expected votes per player from villagers: 4/5 = 0.8 votes
    # - But variance is high
    
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("\nWith 2-vote coordination vs random 1-vote splits:")
    print("- Werewolves give their target a 2-vote head start")
    print("- For a villager to beat this, they need 3+ votes from the 4 villagers")
    print("- Villagers are voting randomly among 5 targets")
    print("- This is VERY unlikely to concentrate 3+ votes on one player")
    
    # Monte Carlo estimate
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION")
    print("="*80)
    
    import random
    random.seed(42)
    
    villagers = ['V1', 'V2', 'V3', 'V4']
    werewolves = ['W1', 'W2']
    all_players = villagers + werewolves
    
    villager_eliminations = 0
    trials = 100000
    
    for _ in range(trials):
        # Werewolves pick a random villager to eliminate
        ww_target = random.choice(villagers)
        
        votes = Counter()
        votes[ww_target] = 2  # Werewolf bloc
        
        # Each villager votes randomly
        for villager in villagers:
            # Vote for any other player
            targets = [p for p in all_players if p != villager]
            vote = random.choice(targets)
            votes[vote] += 1
        
        # Who gets eliminated?
        max_votes = max(votes.values())
        top_candidates = [p for p, v in votes.items() if v == max_votes]
        eliminated = random.choice(top_candidates)  # Random tie-breaking
        
        if eliminated in villagers:
            villager_eliminations += 1
    
    prob = villager_eliminations / trials
    print(f"\nResults from {trials:,} simulated votes:")
    print(f"Villager eliminated: {villager_eliminations:,}/{trials:,} = {100*prob:.2f}%")
    print(f"Werewolf eliminated: {trials - villager_eliminations:,}/{trials:,} = {100*(1-prob):.2f}%")
    
    return prob


def main():
    print("\n" + "="*80)
    print("VOTING PROBABILITY ANALYSIS")
    print("="*80)
    print("\nScenario: 2 Werewolves vs 4 Villagers")
    print("Werewolves: Vote as coordinated bloc (2 votes for same target)")
    print("Villagers: Vote independently/randomly")
    print("="*80)
    
    # Monte Carlo simulation
    monte_carlo_prob = calculate_simplified()
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"\nWith werewolf bloc voting, villagers are eliminated ~{100*monte_carlo_prob:.1f}% of the time")
    print(f"This is {100*monte_carlo_prob/40:.1f}x higher than the 40% random baseline!")
    
    print("\nWhy 70B baseline gets 0% village wins:")
    print("- Werewolves coordinate perfectly → 2-vote bloc")
    print("- Villagers vote randomly → split votes")
    print(f"- Result: ~{100*monte_carlo_prob:.0f}% chance villager eliminated each day vote")
    print("- Over multiple rounds, village NEVER wins")
    
    print("\nWhy 8B baseline gets 40% village wins:")
    print("- 8B villagers must occasionally coordinate by chance")
    print("- Breaking the werewolf bloc advantage sometimes")
    print("- Getting lucky ~40% of the time (matching random baseline)")


if __name__ == "__main__":
    main()
