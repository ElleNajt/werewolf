#!/usr/bin/env python3
"""
Plot probe activation patterns across different stages (prompt, CoT, action).
Analyzes probe scores for villagers vs werewolves across all games in probe conditions.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

def load_game_activations(game_dir: Path) -> Dict:
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

def load_all_activations(base_dirs: List[Path], condition_name: str) -> Dict:
    """Load activations from all games in given directories."""
    all_activations = {
        'villager': {'prompt': [], 'cot': [], 'action': []},
        'werewolf': {'prompt': [], 'cot': [], 'action': []}
    }
    
    games_found = 0
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        
        game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
        
        for game_dir in game_dirs:
            activations = load_game_activations(game_dir)
            if activations:
                games_found += 1
                for role in ['villager', 'werewolf']:
                    for stage in ['prompt', 'cot', 'action']:
                        all_activations[role][stage].extend(activations[role][stage])
    
    print(f"  {condition_name}: Loaded {games_found} games")
    return all_activations

def plot_activation_comparison(conditions_data: Dict[str, Dict], output_file: str):
    """Create comprehensive activation comparison plot."""
    
    stages = ['prompt', 'cot', 'action']
    stage_labels = ['Prompt', 'Chain-of-Thought', 'Action']
    
    n_conditions = len(conditions_data)
    fig, axes = plt.subplots(n_conditions, 3, figsize=(15, 4 * n_conditions))
    
    if n_conditions == 1:
        axes = axes.reshape(1, -1)
    
    for i, (condition_name, data) in enumerate(conditions_data.items()):
        for j, (stage, stage_label) in enumerate(zip(stages, stage_labels)):
            ax = axes[i, j]
            
            villager_scores = data['villager'][stage]
            werewolf_scores = data['werewolf'][stage]
            
            if not villager_scores and not werewolf_scores:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{condition_name}\n{stage_label}')
                continue
            
            # Create histograms
            bins = np.linspace(-75, -65, 30)
            
            if villager_scores:
                ax.hist(villager_scores, bins=bins, alpha=0.6, label='Villagers', 
                       color='#2ecc71', density=True, edgecolor='black', linewidth=0.5)
            if werewolf_scores:
                ax.hist(werewolf_scores, bins=bins, alpha=0.6, label='Werewolves',
                       color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
            
            # Add vertical lines for means
            if villager_scores:
                v_mean = np.mean(villager_scores)
                ax.axvline(v_mean, color='#27ae60', linestyle='--', linewidth=2, 
                          label=f'Villager mean: {v_mean:.1f}')
            if werewolf_scores:
                w_mean = np.mean(werewolf_scores)
                ax.axvline(w_mean, color='#c0392b', linestyle='--', linewidth=2,
                          label=f'Werewolf mean: {w_mean:.1f}')
            
            ax.set_xlabel('Probe Score (more negative = more suspicious)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{condition_name}\n{stage_label}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            # Add sample sizes
            n_v = len(villager_scores)
            n_w = len(werewolf_scores)
            ax.text(0.98, 0.98, f'n_v={n_v}, n_w={n_w}', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()

def plot_stage_comparison(conditions_data: Dict[str, Dict], output_file: str):
    """Create box plot comparing stages within each condition."""
    
    fig, axes = plt.subplots(1, len(conditions_data), figsize=(6 * len(conditions_data), 6))
    
    if len(conditions_data) == 1:
        axes = [axes]
    
    for i, (condition_name, data) in enumerate(conditions_data.items()):
        ax = axes[i]
        
        # Prepare data for box plot
        villager_data = []
        werewolf_data = []
        positions = []
        labels = []
        
        for j, stage in enumerate(['prompt', 'cot', 'action']):
            v_scores = data['villager'][stage]
            w_scores = data['werewolf'][stage]
            
            if v_scores:
                villager_data.append(v_scores)
                positions.append(j * 2)
                labels.append(stage.upper())
            
            if w_scores:
                werewolf_data.append(w_scores)
        
        # Create box plots
        if villager_data:
            bp1 = ax.boxplot(villager_data, positions=[p - 0.3 for p in positions],
                            widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='#2ecc71', alpha=0.6),
                            medianprops=dict(color='#27ae60', linewidth=2),
                            whiskerprops=dict(color='#27ae60'),
                            capprops=dict(color='#27ae60'))
        
        if werewolf_data:
            bp2 = ax.boxplot(werewolf_data, positions=[p + 0.3 for p in positions],
                            widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='#e74c3c', alpha=0.6),
                            medianprops=dict(color='#c0392b', linewidth=2),
                            whiskerprops=dict(color='#c0392b'),
                            capprops=dict(color='#c0392b'))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Probe Score (more negative = more suspicious)', fontsize=11)
        ax.set_title(f'{condition_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.6, label='Villagers'),
            Patch(facecolor='#e74c3c', alpha=0.6, label='Werewolves')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_file}")
    plt.close()

def print_summary_stats(conditions_data: Dict[str, Dict]):
    """Print summary statistics for each condition."""
    print("\n" + "="*80)
    print("PROBE ACTIVATION STATISTICS")
    print("="*80)
    
    for condition_name, data in conditions_data.items():
        print(f"\n{condition_name}:")
        print("-" * 60)
        
        for stage in ['prompt', 'cot', 'action']:
            v_scores = data['villager'][stage]
            w_scores = data['werewolf'][stage]
            
            print(f"\n  {stage.upper()}:")
            if v_scores:
                print(f"    Villagers: mean={np.mean(v_scores):.2f}, std={np.std(v_scores):.2f}, n={len(v_scores)}")
            if w_scores:
                print(f"    Werewolves: mean={np.mean(w_scores):.2f}, std={np.std(w_scores):.2f}, n={len(w_scores)}")
            if v_scores and w_scores:
                diff = np.mean(w_scores) - np.mean(v_scores)
                print(f"    Difference (W-V): {diff:.2f} (negative means werewolves more suspicious)")

def main():
    print("\n" + "="*80)
    print("PROBE ACTIVATION ANALYSIS")
    print("="*80)
    
    # Define conditions to analyze
    conditions = {
        'Probe (Villagers)': [
            Path("../../results/70b_probe_villagers_9622b6b_2025-10-30_17-57-30")
        ],
        'Probe + Warning': [
            Path("../../results/game35"),
            Path("../../results/game36"),
            Path("../../results/game37"),
            Path("../../results/game38"),
            Path("../../results/game39"),
            Path("../../results/game40"),
            Path("../../results/game41"),
            Path("../../results/game42"),
            Path("../../results/game43"),
            Path("../../results/game44"),
            Path("../../results/game45"),
            Path("../../results/game46"),
            Path("../../results/game47"),
            Path("../../results/game48"),
            Path("../../results/game49"),
        ]
    }
    
    # Load all data
    print("\nLoading activation data...")
    conditions_data = {}
    for condition_name, dirs in conditions.items():
        conditions_data[condition_name] = load_all_activations(dirs, condition_name)
    
    # Print summary statistics
    print_summary_stats(conditions_data)
    
    # Create plots
    print("\nCreating visualizations...")
    plot_activation_comparison(conditions_data, "../../results/70b_probe_activations_by_stage.png")
    plot_stage_comparison(conditions_data, "../../results/70b_probe_activations_boxplot.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/70b_probe_activations_by_stage.png")
    print("  - results/70b_probe_activations_boxplot.png")

if __name__ == "__main__":
    main()
