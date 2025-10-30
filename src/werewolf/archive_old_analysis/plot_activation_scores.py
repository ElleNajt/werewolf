#!/usr/bin/env python3
"""
Visualize activation scores from werewolf games.

Creates:
1. Time series plot: All 4 mean scores for each player over time
2. Histograms: Score distributions for werewolves vs villagers (4 plots, one per phase)

Usage:
    python plot_activation_scores.py results/gameN/game_stats.json
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_game_data(json_path: Path):
    """Load game data and extract activation scores by player."""
    with open(json_path) as f:
        data = json.load(f)
    
    players = {p['name']: p['role'] for p in data.get('players', [])}
    player_activations = data.get('player_activations', {})
    
    # Organize data: player -> list of (action_idx, phase_scores)
    player_scores = defaultdict(list)
    
    for player_name, actions in player_activations.items():
        for idx, action in enumerate(actions):
            act = action.get('activations', {})
            scores = {
                'cot_prompt': act.get('cot_prompt_mean_score'),
                'cot_generation': act.get('cot_generation_mean_score'),
                'action_prompt': act.get('action_prompt_mean_score'),
                'action_generation': act.get('action_generation_mean_score'),
            }
            player_scores[player_name].append((idx, scores))
    
    return players, player_scores


def plot_time_series(players, player_scores, output_path):
    """Create time series plot of all 4 scores for each player."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for each phase
    phase_colors = {
        'cot_prompt': '#1f77b4',       # blue
        'cot_generation': '#ff7f0e',   # orange
        'action_prompt': '#2ca02c',    # green
        'action_generation': '#d62728' # red
    }
    
    # Line styles for werewolves vs villagers
    werewolf_style = '-'
    villager_style = '--'
    
    # Plot each player
    for player_name, scores_list in sorted(player_scores.items()):
        role = players.get(player_name, 'unknown')
        linestyle = werewolf_style if role == 'werewolf' else villager_style
        
        # Extract time series for each phase
        indices = [s[0] for s in scores_list]
        
        for phase_name, color in phase_colors.items():
            values = [s[1][phase_name] for s in scores_list]
            # Filter out None values
            valid_indices = [i for i, v in zip(indices, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                label = f"{player_name} ({role}) - {phase_name.replace('_', ' ').title()}"
                ax.plot(valid_indices, valid_values, 
                       color=color, linestyle=linestyle, 
                       marker='o', markersize=4, alpha=0.7,
                       label=label)
    
    ax.set_xlabel('Action Index', fontsize=12)
    ax.set_ylabel('Mean Activation Score', fontsize=12)
    ax.set_title('Activation Scores Over Time by Player and Phase', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Time series plot saved to: {output_path}")
    plt.close()


def plot_histograms(players, player_scores, output_path):
    """Create 4 histograms comparing werewolf vs villager scores for each phase."""
    
    # Collect scores by phase and role
    phase_names = ['cot_prompt', 'cot_generation', 'action_prompt', 'action_generation']
    phase_titles = ['CoT Prompt', 'CoT Generation', 'Action Prompt', 'Action Generation']
    
    scores_by_phase = {phase: {'werewolf': [], 'villager': []} for phase in phase_names}
    
    for player_name, scores_list in player_scores.items():
        role = players.get(player_name, 'unknown')
        if role not in ['werewolf', 'villager']:
            role = 'villager'  # Treat seer as villager for this analysis
        
        for _, scores in scores_list:
            for phase in phase_names:
                value = scores[phase]
                if value is not None:
                    scores_by_phase[phase][role].append(value)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (phase_name, phase_title) in enumerate(zip(phase_names, phase_titles)):
        ax = axes[idx]
        
        werewolf_scores = scores_by_phase[phase_name]['werewolf']
        villager_scores = scores_by_phase[phase_name]['villager']
        
        # Determine bins based on combined data
        all_scores = werewolf_scores + villager_scores
        if all_scores:
            bins = np.linspace(min(all_scores), max(all_scores), 30)
            
            # Plot histograms
            ax.hist(werewolf_scores, bins=bins, alpha=0.6, color='#d62728', 
                   label=f'Werewolf (n={len(werewolf_scores)})', edgecolor='black')
            ax.hist(villager_scores, bins=bins, alpha=0.6, color='#2ca02c', 
                   label=f'Villager (n={len(villager_scores)})', edgecolor='black')
            
            # Add vertical lines for means
            if werewolf_scores:
                ax.axvline(np.mean(werewolf_scores), color='#d62728', 
                          linestyle='--', linewidth=2, label=f'Werewolf mean: {np.mean(werewolf_scores):.2f}')
            if villager_scores:
                ax.axvline(np.mean(villager_scores), color='#2ca02c', 
                          linestyle='--', linewidth=2, label=f'Villager mean: {np.mean(villager_scores):.2f}')
            
            ax.set_xlabel('Mean Activation Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(phase_title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(phase_title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Activation Score Distributions: Werewolves vs Villagers', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Histogram plot saved to: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_activation_scores.py <game_stats.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Load data
    players, player_scores = load_game_data(input_path)
    
    if not player_scores:
        print("No activation scores found in game data")
        sys.exit(1)
    
    # Generate output paths
    output_dir = input_path.parent
    time_series_path = output_dir / "activation_timeseries.png"
    histogram_path = output_dir / "activation_histograms.png"
    
    # Create plots
    plot_time_series(players, player_scores, time_series_path)
    plot_histograms(players, player_scores, histogram_path)
    
    print(f"\nVisualization complete!")
    print(f"  Time series: {time_series_path}")
    print(f"  Histograms: {histogram_path}")


if __name__ == "__main__":
    main()
