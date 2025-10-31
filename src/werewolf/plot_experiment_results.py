#!/usr/bin/env python3
"""
Generate plots for werewolf CoT monitoring experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_all_results():
    """Load results from all experimental conditions."""
    results_dir = Path("../../results")
    
    # Get all final_8b directories
    batch_dirs = sorted(results_dir.glob("final_8b_*"))
    
    results = []
    
    for batch_dir in batch_dirs:
        name = batch_dir.name
        game_dirs = sorted(batch_dir.glob("game*"), key=lambda x: int(x.name.replace("game", "")))
        
        village_wins = 0
        werewolf_wins = 0
        
        for game_dir in game_dirs:
            stats_file = game_dir / "game_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                    winner = stats.get("winner", "")
                    if "village" in winner.lower():
                        village_wins += 1
                    elif "werewolves" in winner.lower() or "werewolf" in winner.lower():
                        werewolf_wins += 1
        
        total = village_wins + werewolf_wins
        if total > 0:
            # Clean up name for display
            display_name = name.split('_d79edc0_')[0].split('_9622b6b_')[0]
            display_name = display_name.replace('final_8b_', '')
            
            results.append({
                'name': display_name,
                'village': village_wins,
                'total': total,
                'pct': 100 * village_wins / total
            })
    
    # Sort by win rate
    results.sort(key=lambda x: x['pct'], reverse=True)
    
    return results

def plot_win_rates(results, output_path="plot_win_rates.png"):
    """Bar plot of village win rates."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    names = [r['name'] for r in results]
    pcts = [r['pct'] for r in results]
    
    # Color bars by category
    colors = []
    for name in names:
        if 'public_cot' in name and 'monitored' not in name:
            colors.append('#2ecc71')  # Green - best
        elif 'probe' in name and 'monitored' not in name and 'werewolf_warning' not in name:
            colors.append('#3498db')  # Blue - probe works
        elif 'baseline' in name:
            colors.append('#95a5a6')  # Gray - baseline
        elif 'werewolf_warning' in name:
            colors.append('#f39c12')  # Orange - interesting finding
        else:
            colors.append('#e74c3c')  # Red - monitoring hurts
    
    bars = ax.barh(names, pcts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, r) in enumerate(zip(bars, results)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{r["village"]}/{r["total"]} ({width:.1f}%)',
                ha='left', va='center', fontweight='bold')
    
    # Add vertical line at baseline
    baseline_pct = next(r['pct'] for r in results if r['name'] == 'baseline')
    ax.axvline(baseline_pct, color='black', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline ({baseline_pct:.0f}%)')
    
    ax.set_xlabel('Village Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=14, fontweight='bold')
    ax.set_title('Werewolf Game: Village Win Rates by Condition\n(2 Werewolves vs 4 Villagers, n=30 games each)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 80)
    ax.legend(fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_monitoring_effect(results, output_path="plot_monitoring_effect.png"):
    """Compare conditions with and without monitoring warnings."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group conditions
    comparisons = [
        ('probe_simple', 'probe_simple_monitored', 'Probe-Simple'),
        ('probe_cautious', 'probe_cautious_monitored', 'Probe-Cautious'),
        ('public_cot', 'public_cot_monitored', 'Public-CoT'),
    ]
    
    x_pos = np.arange(len(comparisons))
    width = 0.35
    
    no_warning = []
    with_warning = []
    labels = []
    
    for no_warn_name, warn_name, label in comparisons:
        no_warn = next((r for r in results if r['name'] == no_warn_name), None)
        warn = next((r for r in results if r['name'] == warn_name), None)
        
        if no_warn and warn:
            no_warning.append(no_warn['pct'])
            with_warning.append(warn['pct'])
            labels.append(label)
    
    x_pos = np.arange(len(labels))
    
    bars1 = ax.bar(x_pos - width/2, no_warning, width, label='No Warning', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, with_warning, width, label='Everyone Warned',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    # Add difference annotations
    for i, (nw, ww) in enumerate(zip(no_warning, with_warning)):
        diff = ww - nw
        y_pos = max(nw, ww) + 5
        ax.text(i, y_pos, f'{diff:+.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='red' if diff < 0 else 'green')
    
    ax.set_ylabel('Village Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Monitoring Warnings on Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_probe_benefit(results, output_path="plot_probe_benefit.png"):
    """Show how probe benefit varies by warning condition."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    baseline = next(r for r in results if r['name'] == 'baseline')['pct']
    
    conditions = [
        ('baseline', 'Baseline\n(No Probe)', '#95a5a6'),
        ('probe_simple', 'Probe\n(No Warning)', '#3498db'),
        ('probe_werewolf_warning', 'Probe\n(WW Warned)', '#f39c12'),
        ('probe_simple_monitored', 'Probe\n(All Warned)', '#e74c3c'),
    ]
    
    names = []
    pcts = []
    colors = []
    
    for cond_name, label, color in conditions:
        r = next((r for r in results if r['name'] == cond_name), None)
        if r:
            names.append(label)
            pcts.append(r['pct'])
            colors.append(color)
    
    bars = ax.bar(names, pcts, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels and delta from baseline
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        if i > 0:  # Not baseline
            delta = pct - baseline
            y_pos = height + 5
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'({delta:+.1f})',
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    color='green' if delta > 0 else 'red')
    
    ax.axhline(baseline, color='black', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Baseline ({baseline:.0f}%)')
    
    ax.set_ylabel('Village Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Probe Benefit Varies by Warning Condition', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_transparency_comparison(results, output_path="plot_transparency_comparison.png"):
    """Compare surveillance (probe) vs transparency (public CoT)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    conditions = [
        ('baseline', 'Baseline\n(Private CoT,\nNo Probe)', '#95a5a6'),
        ('probe_simple', 'Surveillance\n(Private CoT,\nSecret Probe)', '#3498db'),
        ('probe_simple_monitored', 'Surveillance\n(Private CoT,\nKnown Probe)', '#e74c3c'),
        ('public_cot', 'Transparency\n(Public CoT,\nNo Probe)', '#2ecc71'),
    ]
    
    names = []
    pcts = []
    colors = []
    
    for cond_name, label, color in conditions:
        r = next((r for r in results if r['name'] == cond_name), None)
        if r:
            names.append(label)
            pcts.append(r['pct'])
            colors.append(color)
    
    bars = ax.bar(names, pcts, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Village Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Surveillance vs Transparency Approaches', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotations
    ax.annotate('Secret monitoring\nworks well', xy=(1, pcts[1]), xytext=(1.5, pcts[1]+8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.annotate('Monitoring fails\nwhen known', xy=(2, pcts[2]), xytext=(2, pcts[2]-15),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.annotate('Transparency\nworks best!', xy=(3, pcts[3]), xytext=(2.5, pcts[3]+5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_probe_scores(output_path="plot_probe_scores.png"):
    """Compare probe scores across conditions."""
    results_dir = Path("../../results")
    
    conditions = [
        ('final_8b_probe_simple_d79edc0*', 'Probe-Simple\n(No Warning)', '#3498db'),
        ('final_8b_probe_werewolf_warning*', 'Probe-Werewolf-Warning\n(WW Warned)', '#f39c12'),
        ('final_8b_probe_simple_monitored*', 'Probe-Simple-Monitored\n(All Warned)', '#e74c3c'),
    ]
    
    data = []
    
    for pattern, label, color in conditions:
        batch_dirs = list(results_dir.glob(pattern))
        if not batch_dirs:
            continue
        
        batch_dir = batch_dirs[0]
        
        werewolf_scores = []
        villager_scores = []
        
        for game_dir in sorted(batch_dir.glob("game*"), key=lambda x: int(x.name.replace("game", ""))):
            stats_file = game_dir / "game_stats.json"
            if not stats_file.exists():
                continue
            
            with open(stats_file) as f:
                stats = json.load(f)
                
                if "role_reveal_activations" in stats:
                    for player_name, activation in stats["role_reveal_activations"].items():
                        if activation is None:
                            continue
                        
                        player_data = next((p for p in stats["players"] if p["name"] == player_name), None)
                        if not player_data:
                            continue
                        
                        role = player_data["role"]
                        score = activation.get("prompt_mean_score", 0.0)
                        
                        if role == "werewolf":
                            werewolf_scores.append(score)
                        else:
                            villager_scores.append(score)
        
        if werewolf_scores and villager_scores:
            data.append({
                'label': label,
                'color': color,
                'ww_mean': np.mean(werewolf_scores),
                'vil_mean': np.mean(villager_scores),
                'ww_scores': werewolf_scores,
                'vil_scores': villager_scores
            })
    
    if not data:
        print("No probe score data found")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean scores
    x_pos = np.arange(len(data))
    width = 0.35
    
    ww_means = [d['ww_mean'] for d in data]
    vil_means = [d['vil_mean'] for d in data]
    labels = [d['label'] for d in data]
    
    bars1 = ax1.bar(x_pos - width/2, ww_means, width, label='Werewolves',
                    color='#c0392b', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, vil_means, width, label='Villagers',
                    color='#27ae60', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Mean Probe Score (Role-Reveal)', fontsize=12, fontweight='bold')
    ax1.set_title('Probe Scores by Condition', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Separation (difference)
    separations = [d['ww_mean'] - d['vil_mean'] for d in data]
    colors = [d['color'] for d in data]
    
    bars = ax2.bar(labels, separations, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, sep in zip(bars, separations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{sep:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Probe Score Separation\n(Werewolf - Villager)', fontsize=12, fontweight='bold')
    ax2.set_title('Werewolves Still Detectable When Warned', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    """Generate all plots."""
    print("=" * 80)
    print("Generating plots for werewolf CoT monitoring experiments...")
    print("=" * 80)
    print()
    
    # Load results
    results = load_all_results()
    print(f"Loaded {len(results)} experimental conditions")
    print()
    
    # Generate plots
    output_dir = Path("../../results/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_win_rates(results, output_dir / "win_rates.png")
    plot_monitoring_effect(results, output_dir / "monitoring_effect.png")
    plot_probe_benefit(results, output_dir / "probe_benefit.png")
    plot_transparency_comparison(results, output_dir / "transparency_comparison.png")
    plot_probe_scores(output_dir / "probe_scores.png")
    
    print()
    print("=" * 80)
    print(f"✓ All plots saved to: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
