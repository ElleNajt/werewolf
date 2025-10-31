#!/usr/bin/env python3
"""
Plot probe activation patterns across different stages (prompt, CoT, action).
Focused analysis of 70B probe data.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_game_activations(game_dir: Path):
    """Load activation data from a single game."""
    stats_file = game_dir / "game_stats.json"
    if not stats_file.exists():
        return None
    
    with open(stats_file) as f:
        data = json.load(f)
    
    # Extract player activations
    player_activations = data.get('player_activations', {})
    players = data.get('players', [])
    
    # Create role mapping
    role_map = {p['name']: p['role'] for p in players}
    
    # Organize activations by role and stage
    activations = {
        'villager': {'prompt': [], 'cot': [], 'action': []},
        'werewolf': {'prompt': [], 'cot': [], 'action': []}
    }
    
    for player_name, player_acts in player_activations.items():
        role = role_map.get(player_name, 'unknown')
        if role not in ['villager', 'werewolf']:
            continue
        
        for act in player_acts:
            if 'activations' not in act:
                continue
            
            activ = act['activations']
            
            # Extract scores (more negative = more suspicious)
            if 'prompt_score' in activ:
                activations[role]['prompt'].append(activ['prompt_score'])
            if 'cot_score' in activ:
                activations[role]['cot'].append(activ['cot_score'])
            if 'action_score' in activ:
                activations[role]['action'].append(activ['action_score'])
    
    return activations

def load_all_activations(base_dir: Path):
    """Load activations from all games in directory."""
    all_activations = {
        'villager': {'prompt': [], 'cot': [], 'action': []},
        'werewolf': {'prompt': [], 'cot': [], 'action': []}
    }
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return all_activations
    
    game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    games_found = 0
    for game_dir in game_dirs:
        activations = load_game_activations(game_dir)
        if activations:
            games_found += 1
            for role in ['villager', 'werewolf']:
                for stage in ['prompt', 'cot', 'action']:
                    all_activations[role][stage].extend(activations[role][stage])
    
    print(f"Loaded activation data from {games_found} games")
    print(f"  Villagers: {len(all_activations['villager']['prompt'])} prompt, "
          f"{len(all_activations['villager']['cot'])} CoT, "
          f"{len(all_activations['villager']['action'])} action")
    print(f"  Werewolves: {len(all_activations['werewolf']['prompt'])} prompt, "
          f"{len(all_activations['werewolf']['cot'])} CoT, "
          f"{len(all_activations['werewolf']['action'])} action")
    
    return all_activations

def plot_activation_histograms(data, output_file: str):
    """Create histogram plots for each stage."""
    
    stages = ['prompt', 'cot', 'action']
    stage_labels = ['Prompt Reading', 'Chain-of-Thought', 'Action Generation']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (stage, stage_label) in enumerate(zip(stages, stage_labels)):
        ax = axes[i]
        
        villager_scores = data['villager'][stage]
        werewolf_scores = data['werewolf'][stage]
        
        if not villager_scores and not werewolf_scores:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stage_label)
            continue
        
        # Determine bin range
        all_scores = villager_scores + werewolf_scores
        bins = np.linspace(min(all_scores) - 0.5, max(all_scores) + 0.5, 40)
        
        # Create histograms
        if villager_scores:
            ax.hist(villager_scores, bins=bins, alpha=0.7, label='Villagers', 
                   color='#2ecc71', density=True, edgecolor='black', linewidth=0.5)
        if werewolf_scores:
            ax.hist(werewolf_scores, bins=bins, alpha=0.7, label='Werewolves',
                   color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for means
        if villager_scores:
            v_mean = np.mean(villager_scores)
            ax.axvline(v_mean, color='#27ae60', linestyle='--', linewidth=2.5, 
                      label=f'Villager mean: {v_mean:.2f}')
        if werewolf_scores:
            w_mean = np.mean(werewolf_scores)
            ax.axvline(w_mean, color='#c0392b', linestyle='--', linewidth=2.5,
                      label=f'Werewolf mean: {w_mean:.2f}')
        
        ax.set_xlabel('Probe Score (more negative = more suspicious)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{stage_label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample sizes
        n_v = len(villager_scores)
        n_w = len(werewolf_scores)
        ax.text(0.98, 0.98, f'n_villagers={n_v}\nn_werewolves={n_w}', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('70B Probe Activations Across Stages', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHistogram plot saved to: {output_file}")
    plt.close()

def plot_boxplot_comparison(data, output_file: str):
    """Create box plot comparing stages."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for box plot
    villager_data = []
    werewolf_data = []
    positions = []
    labels = ['Prompt', 'CoT', 'Action']
    
    for j, stage in enumerate(['prompt', 'cot', 'action']):
        villager_data.append(data['villager'][stage])
        werewolf_data.append(data['werewolf'][stage])
        positions.append(j)
    
    # Create box plots
    bp1 = ax.boxplot(villager_data, positions=[p - 0.2 for p in positions],
                    widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                    medianprops=dict(color='#27ae60', linewidth=2),
                    whiskerprops=dict(color='#27ae60'),
                    capprops=dict(color='#27ae60'),
                    flierprops=dict(marker='o', markerfacecolor='#2ecc71', markersize=4, alpha=0.5))
    
    bp2 = ax.boxplot(werewolf_data, positions=[p + 0.2 for p in positions],
                    widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                    medianprops=dict(color='#c0392b', linewidth=2),
                    whiskerprops=dict(color='#c0392b'),
                    capprops=dict(color='#c0392b'),
                    flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=4, alpha=0.5))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Probe Score (more negative = more suspicious)', fontsize=12)
    ax.set_title('70B Probe Activations: Villagers vs Werewolves', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Villagers'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Werewolves')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_file}")
    plt.close()

def plot_combined_distribution(data, output_file: str):
    """Create a single plot showing all distributions overlaid."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    colors_v = ['#27ae60', '#16a085', '#2980b9']
    colors_w = ['#c0392b', '#e67e22', '#8e44ad']
    stage_labels = ['Prompt', 'CoT', 'Action']
    
    # Determine overall bin range
    all_scores = []
    for stage in ['prompt', 'cot', 'action']:
        all_scores.extend(data['villager'][stage])
        all_scores.extend(data['werewolf'][stage])
    bins = np.linspace(min(all_scores) - 0.5, max(all_scores) + 0.5, 50)
    
    # Plot each stage
    for i, stage in enumerate(['prompt', 'cot', 'action']):
        v_scores = data['villager'][stage]
        w_scores = data['werewolf'][stage]
        
        if v_scores:
            ax.hist(v_scores, bins=bins, alpha=0.4, label=f'Villagers - {stage_labels[i]}',
                   color=colors_v[i], density=True, histtype='step', linewidth=2)
        if w_scores:
            ax.hist(w_scores, bins=bins, alpha=0.4, label=f'Werewolves - {stage_labels[i]}',
                   color=colors_w[i], density=True, histtype='step', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Probe Score (more negative = more suspicious)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('70B Probe Activations: Combined View Across All Stages', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined distribution plot saved to: {output_file}")
    plt.close()

def print_summary_stats(data):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PROBE ACTIVATION STATISTICS")
    print("="*80)
    
    for stage in ['prompt', 'cot', 'action']:
        v_scores = data['villager'][stage]
        w_scores = data['werewolf'][stage]
        
        print(f"\n{stage.upper()}:")
        if v_scores:
            print(f"  Villagers:  mean={np.mean(v_scores):.3f}, std={np.std(v_scores):.3f}, "
                  f"median={np.median(v_scores):.3f}, n={len(v_scores)}")
        if w_scores:
            print(f"  Werewolves: mean={np.mean(w_scores):.3f}, std={np.std(w_scores):.3f}, "
                  f"median={np.median(w_scores):.3f}, n={len(w_scores)}")
        if v_scores and w_scores:
            diff = np.mean(w_scores) - np.mean(v_scores)
            print(f"  Difference (W-V): {diff:.3f}")
            if diff < 0:
                print(f"  → Werewolves are {abs(diff):.3f} MORE suspicious on average")
            else:
                print(f"  → Werewolves are {diff:.3f} LESS suspicious on average")

def main():
    print("\n" + "="*80)
    print("70B PROBE ACTIVATION ANALYSIS")
    print("="*80)
    
    # Load data from the specific probe directory
    probe_dir = Path("../../results/70b_probe_villagers_9622b6b_2025-10-30_17-57-30")
    
    print(f"\nLoading activation data from:\n  {probe_dir}")
    data = load_all_activations(probe_dir)
    
    # Print summary statistics
    print_summary_stats(data)
    
    # Create plots
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    plot_activation_histograms(data, "../../results/70b_probe_activations_histograms.png")
    plot_boxplot_comparison(data, "../../results/70b_probe_activations_boxplot.png")
    plot_combined_distribution(data, "../../results/70b_probe_activations_combined.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/70b_probe_activations_histograms.png")
    print("  - results/70b_probe_activations_boxplot.png")
    print("  - results/70b_probe_activations_combined.png")
    print("="*80)

if __name__ == "__main__":
    main()
