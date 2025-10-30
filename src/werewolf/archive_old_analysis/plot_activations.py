#!/usr/bin/env python3
"""
Plot histograms of probe activation scores during day phases.
Colored by role: red for werewolves, blue for villagers.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_activations(results_dir: Path):
    """
    Load all probe activations from day phases.

    Returns:
        dict: {role: [scores]} where role is 'werewolf' or 'villager'
    """
    activations = defaultdict(list)

    game_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('game')])

    for game_dir in game_dirs:
        stats_file = game_dir / 'game_stats.json'
        if not stats_file.exists():
            continue

        with open(stats_file) as f:
            data = json.load(f)

        # Skip if no probe enabled
        if not data.get('probe_enabled', False):
            continue

        # Build player role map
        player_roles = {}
        for player in data['players']:
            player_roles[player['name']] = player['role']

        # Extract activations during day phases
        player_activations = data.get('player_activations', {})

        for player_name, acts in player_activations.items():
            role = player_roles.get(player_name, 'unknown')
            if role == 'unknown':
                continue

            for act in acts:
                # Check if this is a day statement (not a vote)
                # Day statements are longer text, votes are just player names
                action_text = act.get('action', '')
                if len(action_text) > 20:  # Heuristic: statements are longer than player names
                    score = act['activations']['aggregate_score']
                    activations[role].append(score)

    return activations


def plot_histogram(ax, activations, title, color_map={'werewolf': 'red', 'villager': 'blue'}):
    """Plot histogram for one condition."""

    # Collect all scores to determine bins
    all_scores = []
    for scores in activations.values():
        all_scores.extend(scores)

    if not all_scores:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Use common bins for all roles
    bins = np.linspace(min(all_scores), max(all_scores), 30)

    # Plot each role
    for role in ['villager', 'werewolf']:
        if role in activations and activations[role]:
            ax.hist(activations[role], bins=bins, alpha=0.6,
                   color=color_map[role], label=f'{role.capitalize()} (n={len(activations[role])})',
                   edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Probe Score (higher = more deceptive)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add mean lines
    for role in ['villager', 'werewolf']:
        if role in activations and activations[role]:
            mean_score = np.mean(activations[role])
            ax.axvline(mean_score, color=color_map[role], linestyle='--', linewidth=2, alpha=0.8,
                      label=f'{role.capitalize()} mean: {mean_score:.2f}')


def main():
    repo_root = Path(__file__).parent.parent.parent

    # Define experiments to plot
    experiments = [
        ('experiment_treatment', 'Treatment\n(WITH Role Context)\nVillage Win Rate: 100%'),
        ('experiment_ablation_6player', 'Ablation\n(WITHOUT Role Context, "higher=deceptive")\nVillage Win Rate: 0%'),
        ('experiment_outlier', 'Outlier Framing\n(WITHOUT Role Context, "look for outliers")\nVillage Win Rate: 10%'),
        ('experiment_inverted', 'Inverted Instructions\n(WITHOUT Role Context, "lower=deceptive")\nVillage Win Rate: 0%'),
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Probe Activation Scores During Day Phase Statements\nComparing Role Context and Instruction Framing',
                 fontsize=14, fontweight='bold')

    for ax, (exp_name, title) in zip(axes, experiments):
        results_dir = repo_root / 'results' / exp_name

        if not results_dir.exists():
            ax.text(0.5, 0.5, f'No data\n({exp_name})', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        activations = load_activations(results_dir)

        if not activations:
            ax.text(0.5, 0.5, 'No probe data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        plot_histogram(ax, activations, title)

    plt.tight_layout()

    # Save figure
    output_file = repo_root / 'results' / 'activation_distributions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Also show statistics
    print("\nStatistics:")
    print("="*70)
    for exp_name, title in experiments:
        results_dir = repo_root / 'results' / exp_name
        if not results_dir.exists():
            continue

        activations = load_activations(results_dir)
        if not activations:
            continue

        print(f"\n{title}:")
        for role in ['villager', 'werewolf']:
            if role in activations and activations[role]:
                scores = activations[role]
                print(f"  {role.capitalize()}:")
                print(f"    n={len(scores)}")
                print(f"    mean={np.mean(scores):.3f}")
                print(f"    std={np.std(scores):.3f}")
                print(f"    median={np.median(scores):.3f}")

        if 'villager' in activations and 'werewolf' in activations:
            if activations['villager'] and activations['werewolf']:
                separation = np.mean(activations['werewolf']) - np.mean(activations['villager'])
                print(f"  Separation (werewolf - villager): {separation:.3f}")


if __name__ == '__main__':
    main()
