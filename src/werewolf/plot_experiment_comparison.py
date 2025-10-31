#!/usr/bin/env python3
"""
Create summary plots comparing experimental conditions for 70B and 8B models.
Compares: Baseline, Public CoT, and Probe conditions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_experiment_results(base_dir: Path) -> Dict:
    """Load game results from all games in a directory."""
    results = {
        'games': 0,
        'village_wins': 0,
        'werewolf_wins': 0,
        'game_lengths': []
    }
    
    if not base_dir.exists():
        print(f"Warning: {base_dir} not found")
        return results
    
    game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    for game_dir in game_dirs:
        stats_file = game_dir / 'game_stats.json'
        if not stats_file.exists():
            continue
        
        with open(stats_file) as f:
            data = json.load(f)
        
        results['games'] += 1
        winner = data.get('winner', '').lower()
        if 'village' in winner or 'villager' in winner:
            results['village_wins'] += 1
        elif 'werewolf' in winner or 'werewolves' in winner:
            results['werewolf_wins'] += 1
        
        results['game_lengths'].append(data.get('total_turns', 0))
    
    return results


def create_comparison_plots(model_size: str, conditions: Dict[str, Path], output_file: Path):
    """Create comparison plots for a model size."""
    
    # Load data for each condition
    data = {}
    for name, path in conditions.items():
        data[name] = load_experiment_results(path)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    condition_names = list(data.keys())
    x_pos = np.arange(len(condition_names))
    
    village_win_rates = []
    werewolf_win_rates = []
    n_games = []
    
    for name in condition_names:
        total = data[name]['games']
        n_games.append(total)
        if total > 0:
            village_win_rates.append(100 * data[name]['village_wins'] / total)
            werewolf_win_rates.append(100 * data[name]['werewolf_wins'] / total)
        else:
            village_win_rates.append(0)
            werewolf_win_rates.append(0)
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, village_win_rates, width, 
                    label='Village Wins', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, werewolf_win_rates, width,
                    label='Werewolf Wins', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar1, bar2, n) in enumerate(zip(bars1, bars2, n_games)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Add n= label below
        ax.text(i, -12, f'n={n}', ha='center', va='top', fontsize=10, style='italic')
    
    ax.set_ylabel('Win Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_size} Model: Win Rates by Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_names, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='50% (balanced)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{model_size} MODEL SUMMARY:")
    print("=" * 80)
    for name in condition_names:
        d = data[name]
        total = d['games']
        if total > 0:
            v_rate = 100 * d['village_wins'] / total
            w_rate = 100 * d['werewolf_wins'] / total
            avg_len = np.mean(d['game_lengths']) if d['game_lengths'] else 0
            print(f"\n{name}:")
            print(f"  Games: {total}")
            print(f"  Village win rate: {v_rate:.1f}%")
            print(f"  Werewolf win rate: {w_rate:.1f}%")
            print(f"  Avg game length: {avg_len:.1f} turns")


def create_cross_model_comparison(data_70b: Dict, data_8b: Dict, output_file: Path):
    """Create side-by-side comparison of 70B and 8B models."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    models = ['70B', '8B']
    all_data = [data_70b, data_8b]
    
    for idx, (model, model_data, ax) in enumerate(zip(models, all_data, axes)):
        condition_names = list(model_data.keys())
        x_pos = np.arange(len(condition_names))
        
        village_win_rates = []
        n_games = []
        
        for name in condition_names:
            total = model_data[name]['games']
            n_games.append(total)
            if total > 0:
                village_win_rates.append(100 * model_data[name]['village_wins'] / total)
            else:
                village_win_rates.append(0)
        
        bars = ax.bar(x_pos, village_win_rates, color='#3498db', alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, n) in enumerate(zip(bars, n_games)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
            ax.text(i, -8, f'n={n}', ha='center', va='top', fontsize=9, style='italic')
        
        ax.set_ylabel('Village Win Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model} Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_names, fontsize=10, rotation=15, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1, 
                  label='50% (balanced)')
        ax.legend(fontsize=9)
    
    plt.suptitle('Village Win Rates: 70B vs 8B Model Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON PLOTS")
    print("=" * 80)
    
    results_dir = Path("../../results")
    
    # 70B conditions
    conditions_70b = {
        'Baseline': results_dir / '70b_baseline_9622b6b_2025-10-30_17-57-30',
        'Public CoT': results_dir / '70b_public_cot_9622b6b_2025-10-30_17-57-30',
        'Probe\n(Villagers)': results_dir / '70b_probe_villagers_9622b6b_2025-10-30_17-57-30',
    }
    
    # 8B conditions
    conditions_8b = {
        'Baseline': results_dir / 'final_8b_baseline_d79edc0_f671a91',
        'Public CoT': results_dir / 'final_8b_public_cot_d79edc0_3b87d87',
        'Probe\n(Simple)': results_dir / 'final_8b_probe_simple_d79edc0_8288738',
    }
    
    # Create 70B comparison
    print("\nProcessing 70B data...")
    data_70b = {}
    for name, path in conditions_70b.items():
        data_70b[name] = load_experiment_results(path)
    
    create_comparison_plots('70B', conditions_70b, 
                           results_dir / '70b_experiment_comparison.png')
    
    # Create 8B comparison
    print("\nProcessing 8B data...")
    data_8b = {}
    for name, path in conditions_8b.items():
        data_8b[name] = load_experiment_results(path)
    
    create_comparison_plots('8B', conditions_8b,
                           results_dir / '8b_experiment_comparison.png')
    
    # Create cross-model comparison
    print("\nCreating cross-model comparison...")
    create_cross_model_comparison(data_70b, data_8b,
                                 results_dir / 'cross_model_comparison.png')
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/70b_experiment_comparison.png")
    print("  - results/8b_experiment_comparison.png")
    print("  - results/cross_model_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
