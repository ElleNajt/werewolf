#!/usr/bin/env python3
"""
Generate cumulative visualizations across all games in each condition.
- Cumulative activation histograms (union of all games)
- Aggregate distrust time series (mean across all games)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def load_all_activations(batch_dir):
    """Load all role-reveal and gameplay activations from all games in a batch."""
    role_reveal_ww = []
    role_reveal_vil = []
    gameplay_ww = []
    gameplay_vil = []
    
    for game_dir in sorted(batch_dir.glob("game*"), key=lambda x: int(x.name.replace("game", ""))):
        stats_file = game_dir / "game_stats.json"
        if not stats_file.exists():
            continue
        
        with open(stats_file) as f:
            data = json.load(f)
            
            # Get role-reveal activations
            if "role_reveal_activations" in data:
                for player in data["players"]:
                    name = player["name"]
                    role = player["role"]
                    
                    if name in data["role_reveal_activations"]:
                        activation = data["role_reveal_activations"][name]
                        if activation and "prompt_mean_score" in activation:
                            score = activation["prompt_mean_score"]
                            if role == "werewolf":
                                role_reveal_ww.append(score)
                            else:
                                role_reveal_vil.append(score)
            
            # Get gameplay activations (all actions during game)
            if "player_activations" in data:
                for player in data["players"]:
                    name = player["name"]
                    role = player["role"]
                    
                    if name in data["player_activations"]:
                        for action in data["player_activations"][name]:
                            if "activations" in action and "aggregate_score" in action["activations"]:
                                score = action["activations"]["aggregate_score"]
                                if role == "werewolf":
                                    gameplay_ww.append(score)
                                else:
                                    gameplay_vil.append(score)
    
    return {
        'role_reveal_ww': role_reveal_ww,
        'role_reveal_vil': role_reveal_vil,
        'gameplay_ww': gameplay_ww,
        'gameplay_vil': gameplay_vil
    }


def plot_cumulative_histograms(conditions, output_dir):
    """Plot cumulative activation histograms for each condition."""
    
    for pattern, name in conditions:
        results_dir = Path("../../results")
        batch_dirs = list(results_dir.glob(pattern))
        
        if not batch_dirs:
            continue
        
        batch_dir = batch_dirs[0]
        activations = load_all_activations(batch_dir)
        
        # Create 2-panel figure: role-reveal and gameplay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel 1: Role-reveal activations
        if activations['role_reveal_ww'] and activations['role_reveal_vil']:
            bins = np.linspace(
                min(min(activations['role_reveal_ww']), min(activations['role_reveal_vil'])),
                max(max(activations['role_reveal_ww']), max(activations['role_reveal_vil'])),
                30
            )
            
            ax1.hist(activations['role_reveal_vil'], bins=bins, alpha=0.6, 
                    label=f'Villagers (n={len(activations["role_reveal_vil"])})',
                    color='#27ae60', edgecolor='black')
            ax1.hist(activations['role_reveal_ww'], bins=bins, alpha=0.6,
                    label=f'Werewolves (n={len(activations["role_reveal_ww"])})',
                    color='#c0392b', edgecolor='black')
            
            ax1.axvline(np.mean(activations['role_reveal_vil']), color='#27ae60', 
                       linestyle='--', linewidth=2, label=f'Vil mean: {np.mean(activations["role_reveal_vil"]):.2f}')
            ax1.axvline(np.mean(activations['role_reveal_ww']), color='#c0392b',
                       linestyle='--', linewidth=2, label=f'WW mean: {np.mean(activations["role_reveal_ww"]):.2f}')
            
            ax1.set_xlabel('Probe Score (higher = more suspicious)', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title(f'Role-Reveal Activations\n(100% accuracy: WW > Vil)', fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # Panel 2: Gameplay activations
        if activations['gameplay_ww'] and activations['gameplay_vil']:
            bins = np.linspace(
                min(min(activations['gameplay_ww']), min(activations['gameplay_vil'])),
                max(max(activations['gameplay_ww']), max(activations['gameplay_vil'])),
                40
            )
            
            ax2.hist(activations['gameplay_vil'], bins=bins, alpha=0.6,
                    label=f'Villagers (n={len(activations["gameplay_vil"])})',
                    color='#27ae60', edgecolor='black')
            ax2.hist(activations['gameplay_ww'], bins=bins, alpha=0.6,
                    label=f'Werewolves (n={len(activations["gameplay_ww"])})',
                    color='#c0392b', edgecolor='black')
            
            ax2.axvline(np.mean(activations['gameplay_vil']), color='#27ae60',
                       linestyle='--', linewidth=2, label=f'Vil mean: {np.mean(activations["gameplay_vil"]):.2f}')
            ax2.axvline(np.mean(activations['gameplay_ww']), color='#c0392b',
                       linestyle='--', linewidth=2, label=f'WW mean: {np.mean(activations["gameplay_ww"]):.2f}')
            
            ax2.set_xlabel('Probe Score (higher = more suspicious)', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title(f'Gameplay Activations\n(0% accuracy: Complete overlap)', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        fig.suptitle(f'{name} - Cumulative Activation Distributions\n(All games combined)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / f'cumulative_histograms_{name.lower().replace("-", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def load_game_timeseries(batch_dir):
    """Load cumulative distrust time series for all games."""
    all_timeseries = defaultdict(list)  # {player_name: [cumulative_scores across games]}
    
    for game_dir in sorted(batch_dir.glob("game*"), key=lambda x: int(x.name.replace("game", ""))):
        stats_file = game_dir / "game_stats.json"
        if not stats_file.exists():
            continue
        
        with open(stats_file) as f:
            data = json.load(f)
            
            # Get player roles
            player_roles = {p["name"]: p["role"] for p in data["players"]}
            
            # Track cumulative scores for each player in this game
            game_cumulative = defaultdict(list)
            game_scores = defaultdict(float)
            
            if "player_activations" in data:
                # Build time series of cumulative scores
                for player_name in player_roles:
                    if player_name in data["player_activations"]:
                        for action in data["player_activations"][player_name]:
                            if "activations" in action and "aggregate_score" in action["activations"]:
                                score = action["activations"]["aggregate_score"]
                                game_scores[player_name] += score
                                game_cumulative[player_name].append(game_scores[player_name])
                
                # Store this game's timeseries
                for player_name, role in player_roles.items():
                    if player_name in game_cumulative and len(game_cumulative[player_name]) > 0:
                        # Pad to same length if needed (some players may have fewer actions)
                        all_timeseries[(role, len(game_cumulative[player_name]))].append(
                            game_cumulative[player_name]
                        )
    
    return all_timeseries


def plot_aggregate_timeseries(conditions, output_dir):
    """Plot aggregate cumulative distrust time series."""
    
    fig, axes = plt.subplots(len(conditions), 1, figsize=(14, 6*len(conditions)))
    if len(conditions) == 1:
        axes = [axes]
    
    for ax, (pattern, name) in zip(axes, conditions):
        results_dir = Path("../../results")
        batch_dirs = list(results_dir.glob(pattern))
        
        if not batch_dirs:
            continue
        
        batch_dir = batch_dirs[0]
        timeseries = load_game_timeseries(batch_dir)
        
        # Aggregate by role
        ww_series = []
        vil_series = []
        
        for (role, length), series_list in timeseries.items():
            if role == "werewolf":
                ww_series.extend(series_list)
            else:
                vil_series.extend(series_list)
        
        # Find common length (most common series length)
        if ww_series and vil_series:
            # Take minimum length to align all series
            min_len = min(min(len(s) for s in ww_series), min(len(s) for s in vil_series))
            
            # Truncate all to same length
            ww_series = [s[:min_len] for s in ww_series]
            vil_series = [s[:min_len] for s in vil_series]
            
            # Calculate mean and std
            ww_mean = np.mean(ww_series, axis=0)
            ww_std = np.std(ww_series, axis=0)
            vil_mean = np.mean(vil_series, axis=0)
            vil_std = np.std(vil_series, axis=0)
            
            x = np.arange(len(ww_mean))
            
            # Plot with confidence bands
            ax.plot(x, ww_mean, color='#c0392b', linewidth=3, label=f'Werewolves (n={len(ww_series)})')
            ax.fill_between(x, ww_mean - ww_std, ww_mean + ww_std, color='#c0392b', alpha=0.2)
            
            ax.plot(x, vil_mean, color='#27ae60', linewidth=3, label=f'Villagers (n={len(vil_series)})')
            ax.fill_between(x, vil_mean - vil_std, vil_mean + vil_std, color='#27ae60', alpha=0.2)
            
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            
            ax.set_xlabel('Action Number (chronological)', fontweight='bold', fontsize=12)
            ax.set_ylabel('Cumulative Probe Score', fontweight='bold', fontsize=12)
            ax.set_title(f'{name}\nMean Cumulative Suspicion Over Time', fontweight='bold', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'aggregate_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comparison_panel(conditions, output_dir):
    """Create a comparison panel showing all conditions side-by-side."""
    
    fig, axes = plt.subplots(2, len(conditions), figsize=(6*len(conditions), 12))
    
    for col_idx, (pattern, name) in enumerate(conditions):
        results_dir = Path("../../results")
        batch_dirs = list(results_dir.glob(pattern))
        
        if not batch_dirs:
            continue
        
        batch_dir = batch_dirs[0]
        activations = load_all_activations(batch_dir)
        
        # Row 1: Role-reveal histogram
        ax1 = axes[0, col_idx] if len(conditions) > 1 else axes[0]
        
        if activations['role_reveal_ww'] and activations['role_reveal_vil']:
            bins = np.linspace(
                min(min(activations['role_reveal_ww']), min(activations['role_reveal_vil'])),
                max(max(activations['role_reveal_ww']), max(activations['role_reveal_vil'])),
                20
            )
            
            ax1.hist(activations['role_reveal_vil'], bins=bins, alpha=0.6,
                    label='Villagers', color='#27ae60', edgecolor='black')
            ax1.hist(activations['role_reveal_ww'], bins=bins, alpha=0.6,
                    label='Werewolves', color='#c0392b', edgecolor='black')
            
            ax1.set_xlabel('Probe Score', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title(f'{name}\nRole-Reveal (100% acc)', fontweight='bold', fontsize=12)
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # Row 2: Gameplay histogram
        ax2 = axes[1, col_idx] if len(conditions) > 1 else axes[1]
        
        if activations['gameplay_ww'] and activations['gameplay_vil']:
            bins = np.linspace(
                min(min(activations['gameplay_ww']), min(activations['gameplay_vil'])),
                max(max(activations['gameplay_ww']), max(activations['gameplay_vil'])),
                30
            )
            
            ax2.hist(activations['gameplay_vil'], bins=bins, alpha=0.6,
                    label='Villagers', color='#27ae60', edgecolor='black')
            ax2.hist(activations['gameplay_ww'], bins=bins, alpha=0.6,
                    label='Werewolves', color='#c0392b', edgecolor='black')
            
            ax2.set_xlabel('Probe Score', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title(f'Gameplay (0% acc)', fontweight='bold', fontsize=12)
            ax2.legend()
            ax2.grid(alpha=0.3)
    
    plt.suptitle('Probe Accuracy: Role-Reveal vs Gameplay', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'comparison_panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all cumulative visualizations."""
    print("=" * 80)
    print("Generating cumulative visualizations...")
    print("=" * 80)
    print()
    
    output_dir = Path("../../results/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    conditions = [
        ('final_8b_probe_simple_d79edc0*', 'Probe-Simple'),
        ('final_8b_probe_werewolf_warning*', 'Probe-WW-Warning'),
        ('final_8b_probe_simple_monitored*', 'Probe-Monitored'),
    ]
    
    print("Generating cumulative histograms for each condition...")
    plot_cumulative_histograms(conditions, output_dir)
    print()
    
    print("Generating aggregate time series...")
    plot_aggregate_timeseries(conditions, output_dir)
    print()
    
    print("Generating comparison panel...")
    plot_comparison_panel(conditions, output_dir)
    print()
    
    print("=" * 80)
    print("✓ All cumulative visualizations generated")
    print("=" * 80)


if __name__ == "__main__":
    main()
