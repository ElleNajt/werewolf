#!/usr/bin/env python3
"""
Analyze results from Werewolf experiment comparing control vs treatment conditions.
"""

import json
from pathlib import Path
from typing import Dict, List
import scipy.stats


def count_werewolves_eliminated(game_stats: Dict) -> int:
    """Count how many werewolves were eliminated in this game."""
    count = 0
    for player in game_stats['players']:
        if not player.get('survived', True) and player['role'] == 'werewolf':
            count += 1
    return count


def load_game_results(results_dir: Path) -> List[Dict]:
    """Load all game_stats.json files from a results directory."""
    games = []

    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist")
        return games

    for game_dir in sorted(results_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        stats_file = game_dir / "game_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                games.append(json.load(f))

    return games


def analyze_condition(games: List[Dict], condition_name: str) -> Dict:
    """Analyze games for a single condition."""

    if not games:
        return {
            'n_games': 0,
            'villager_wins': 0,
            'villager_win_rate': 0.0,
            'avg_turns': 0.0,
            'werewolves_eliminated': []
        }

    villager_wins = sum(1 for g in games if g['winner'] == 'Village')
    total_turns = sum(g['total_turns'] for g in games)
    werewolves_eliminated = [count_werewolves_eliminated(g) for g in games]

    return {
        'n_games': len(games),
        'villager_wins': villager_wins,
        'villager_win_rate': villager_wins / len(games) if games else 0.0,
        'avg_turns': total_turns / len(games) if games else 0.0,
        'werewolves_eliminated': werewolves_eliminated
    }


def fishers_exact_test(control_wins, control_total, treatment_wins, treatment_total):
    """Perform Fisher's exact test for win rates."""
    # Create contingency table
    # Rows: [wins, losses], Cols: [control, treatment]
    table = [
        [control_wins, treatment_wins],
        [control_total - control_wins, treatment_total - treatment_wins]
    ]

    oddsratio, p_value = scipy.stats.fisher_exact(table)
    return oddsratio, p_value


def main():
    """Analyze experimental results."""

    # Paths to result directories
    control_dir = Path("../../results/experiment_control")
    treatment_dir = Path("../../results/experiment_treatment")

    print(f"""
{'='*70}
WEREWOLF EXPERIMENT RESULTS ANALYSIS
{'='*70}

Loading results from:
  Control: {control_dir}
  Treatment: {treatment_dir}
""")

    # Load game results
    control_games = load_game_results(control_dir)
    treatment_games = load_game_results(treatment_dir)

    # Analyze each condition
    control_stats = analyze_condition(control_games, "control")
    treatment_stats = analyze_condition(treatment_games, "treatment")

    print(f"""
{'='*70}
DESCRIPTIVE STATISTICS
{'='*70}

CONTROL CONDITION (No probe scores shown)
  Games played: {control_stats['n_games']}
  Villager wins: {control_stats['villager_wins']}
  Villager win rate: {control_stats['villager_win_rate']:.1%}
  Average turns: {control_stats['avg_turns']:.1f}

  Werewolves eliminated distribution:
    0 werewolves: {control_stats['werewolves_eliminated'].count(0)} games
    1 werewolf:   {control_stats['werewolves_eliminated'].count(1)} games
    2 werewolves: {control_stats['werewolves_eliminated'].count(2)} games


TREATMENT CONDITION (Probe scores shown with relative explanation)
  Games played: {treatment_stats['n_games']}
  Villager wins: {treatment_stats['villager_wins']}
  Villager win rate: {treatment_stats['villager_win_rate']:.1%}
  Average turns: {treatment_stats['avg_turns']:.1f}

  Werewolves eliminated distribution:
    0 werewolves: {treatment_stats['werewolves_eliminated'].count(0)} games
    1 werewolf:   {treatment_stats['werewolves_eliminated'].count(1)} games
    2 werewolves: {treatment_stats['werewolves_eliminated'].count(2)} games
""")

    # Statistical tests
    if control_stats['n_games'] > 0 and treatment_stats['n_games'] > 0:
        print(f"""
{'='*70}
STATISTICAL TESTS
{'='*70}
""")

        # Fisher's exact test for win rates
        oddsratio, p_value = fishers_exact_test(
            control_stats['villager_wins'],
            control_stats['n_games'],
            treatment_stats['villager_wins'],
            treatment_stats['n_games']
        )

        print(f"""
Fisher's Exact Test (Villager Win Rate)
  Odds ratio: {oddsratio:.3f}
  p-value: {p_value:.4f}
  Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}
""")

        # Mann-Whitney U test for turns
        u_stat, p_value_turns = scipy.stats.mannwhitneyu(
            [g['total_turns'] for g in control_games],
            [g['total_turns'] for g in treatment_games],
            alternative='two-sided'
        )

        print(f"""
Mann-Whitney U Test (Number of Turns)
  U statistic: {u_stat:.1f}
  p-value: {p_value_turns:.4f}
  Significant at α=0.05: {'YES' if p_value_turns < 0.05 else 'NO'}
""")

        # Mann-Whitney U test for werewolves eliminated
        u_stat_ww, p_value_ww = scipy.stats.mannwhitneyu(
            control_stats['werewolves_eliminated'],
            treatment_stats['werewolves_eliminated'],
            alternative='two-sided'
        )

        print(f"""
Mann-Whitney U Test (Werewolves Eliminated)
  U statistic: {u_stat_ww:.1f}
  p-value: {p_value_ww:.4f}
  Significant at α=0.05: {'YES' if p_value_ww < 0.05 else 'NO'}
""")

    # Summary
    print(f"""
{'='*70}
SUMMARY
{'='*70}

Treatment effect on villager win rate:
  Control: {control_stats['villager_win_rate']:.1%}
  Treatment: {treatment_stats['villager_win_rate']:.1%}
  Difference: {(treatment_stats['villager_win_rate'] - control_stats['villager_win_rate']):.1%}

Treatment effect on average turns:
  Control: {control_stats['avg_turns']:.1f} turns
  Treatment: {treatment_stats['avg_turns']:.1f} turns
  Difference: {(treatment_stats['avg_turns'] - control_stats['avg_turns']):.1f} turns

Treatment effect on werewolves eliminated (average):
  Control: {sum(control_stats['werewolves_eliminated']) / len(control_stats['werewolves_eliminated']) if control_stats['werewolves_eliminated'] else 0:.2f} werewolves
  Treatment: {sum(treatment_stats['werewolves_eliminated']) / len(treatment_stats['werewolves_eliminated']) if treatment_stats['werewolves_eliminated'] else 0:.2f} werewolves
  Difference: {(sum(treatment_stats['werewolves_eliminated']) / len(treatment_stats['werewolves_eliminated']) if treatment_stats['werewolves_eliminated'] else 0) - (sum(control_stats['werewolves_eliminated']) / len(control_stats['werewolves_eliminated']) if control_stats['werewolves_eliminated'] else 0):.2f} werewolves

{'='*70}
""")


if __name__ == "__main__":
    main()
