#!/usr/bin/env python3
"""Comprehensive analysis of all 8B experiments."""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import math

def analyze_game_directory(game_dir: Path):
    """Analyze a single game directory."""
    stats_file = game_dir / "game_stats.json"
    if not stats_file.exists():
        return None
        
    with open(stats_file) as f:
        data = json.load(f)
    
    return {
        'winner': data['winner'],
        'turns': data['total_turns'],
        'probe_enabled': data.get('probe_enabled', False),
    }

def analyze_condition(base_dir: Path, condition_name: str):
    """Analyze all games in a condition directory."""
    games = []
    
    if not base_dir.exists():
        print(f"Warning: {base_dir} not found")
        return None
    
    # Find all game subdirectories
    game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    for game_dir in game_dirs:
        result = analyze_game_directory(game_dir)
        if result:
            games.append(result)
    
    if not games:
        return None
    
    village_wins = sum(1 for g in games if g['winner'] == 'Village')
    total_games = len(games)
    avg_turns = np.mean([g['turns'] for g in games])
    
    return {
        'condition': condition_name,
        'total_games': total_games,
        'village_wins': village_wins,
        'werewolf_wins': total_games - village_wins,
        'village_win_rate': village_wins / total_games if total_games > 0 else 0,
        'avg_turns': avg_turns,
    }

def factorial(n):
    """Calculate factorial."""
    return math.factorial(n)

def fisher_exact_test(a, b, c, d):
    """Fisher's exact test for 2x2 contingency table."""
    def hypergeometric_prob(a, b, c, d):
        n = a + b + c + d
        return (factorial(a+b) * factorial(c+d) * factorial(a+c) * factorial(b+d)) / \
               (factorial(n) * factorial(a) * factorial(b) * factorial(c) * factorial(d))
    
    cutoff = hypergeometric_prob(a, b, c, d)
    p_value = 0.0
    n1 = a + b
    n2 = c + d
    
    for i in range(max(0, n1 - (b + d)), min(n1, a + c) + 1):
        j = n1 - i
        k = (a + c) - i
        l = n2 - k
        
        if j >= 0 and k >= 0 and l >= 0:
            prob = hypergeometric_prob(i, j, k, l)
            if prob <= cutoff + 1e-10:
                p_value += prob
    
    return min(p_value, 1.0)

def calculate_pairwise_pvalues(results: List[Dict]):
    """Calculate pairwise p-values using Fisher's exact test."""
    pvalues = {}
    
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i >= j:
                continue
            
            a = r1['village_wins']
            b = r1['werewolf_wins']
            c = r2['village_wins']
            d = r2['werewolf_wins']
            
            pvalue = fisher_exact_test(a, b, c, d)
            
            key = f"{r1['condition']} vs {r2['condition']}"
            pvalues[key] = {
                'p_value': pvalue,
                'significant': pvalue < 0.05,
                'highly_significant': pvalue < 0.01
            }
    
    return pvalues

def create_summary_table(results: List[Dict]):
    """Create a summary table as formatted string."""
    for r in results:
        r['werewolf_win_rate'] = 1 - r['village_win_rate']
    
    header = f"{'Condition':<30} {'Games':>6} {'V-Wins':>7} {'W-Wins':>7} {'V-Rate':>8} {'W-Rate':>8} {'Avg-Turns':>10}"
    separator = "=" * len(header)
    
    rows = [separator, header, separator]
    for r in results:
        row = (f"{r['condition']:<30} {r['total_games']:>6} {r['village_wins']:>7} "
               f"{r['werewolf_wins']:>7} {r['village_win_rate']:>7.1%} "
               f"{r['werewolf_win_rate']:>7.1%} {r['avg_turns']:>10.1f}")
        rows.append(row)
    rows.append(separator)
    
    return "\n".join(rows)

def plot_results(results: List[Dict], output_file: str):
    """Create visualization of results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    conditions = [r['condition'] for r in results]
    village_rates = [r['village_win_rate'] * 100 for r in results]
    werewolf_rates = [100 - v for v in village_rates]
    avg_turns = [r['avg_turns'] for r in results]
    
    # Plot 1: Win rates
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, village_rates, width, label='Village Win Rate', color='#2ecc71')
    ax1.bar(x + width/2, werewolf_rates, width, label='Werewolf Win Rate', color='#e74c3c')
    
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('8B Model: Win Rates by Condition', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (v, w) in enumerate(zip(village_rates, werewolf_rates)):
        ax1.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, w + 2, f'{w:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Average turns
    ax2.bar(x, avg_turns, color='#3498db')
    ax2.set_ylabel('Average Turns', fontsize=12)
    ax2.set_title('8B Model: Average Game Length', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, turns in enumerate(avg_turns):
        ax2.text(i, turns + 0.05, f'{turns:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()

def main():
    print("\n" + "="*80)
    print("8B MODEL: COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("="*80)
    
    results = []
    
    # Analyze experiments
    baseline = analyze_condition(
        Path("../../results/final_8b_baseline_d79edc0_f671a91"),
        "Baseline"
    )
    if baseline:
        results.append(baseline)
    
    public_cot = analyze_condition(
        Path("../../results/final_8b_public_cot_d79edc0_3b87d87"),
        "Public CoT"
    )
    if public_cot:
        results.append(public_cot)
    
    probe_simple = analyze_condition(
        Path("../../results/final_8b_probe_simple_d79edc0_8288738"),
        "Probe (Simple)"
    )
    if probe_simple:
        results.append(probe_simple)
    
    probe_cautious = analyze_condition(
        Path("../../results/final_8b_probe_cautious_d79edc0_8c121dc"),
        "Probe (Cautious)"
    )
    if probe_cautious:
        results.append(probe_cautious)
    
    if not results:
        print("ERROR: No results found!")
        return
    
    # Create summary table
    table = create_summary_table(results)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(table)
    
    # Save to CSV
    csv_file = "../../results/8b_summary.csv"
    with open(csv_file, 'w') as f:
        f.write("Condition,Total_Games,Village_Wins,Werewolf_Wins,Village_Win_Rate,Werewolf_Win_Rate,Avg_Turns\n")
        for r in results:
            f.write(f"{r['condition']},{r['total_games']},{r['village_wins']},{r['werewolf_wins']},"
                   f"{r['village_win_rate']:.4f},{r['werewolf_win_rate']:.4f},{r['avg_turns']:.2f}\n")
    print(f"\nTable saved to: {csv_file}")
    
    # Create plots
    plot_file = "../../results/8b_comparison.png"
    plot_results(results, plot_file)
    
    # Calculate p-values
    pvalues = calculate_pairwise_pvalues(results)
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (Fisher's Exact Test)")
    print("="*80)
    for comparison, stats in sorted(pvalues.items()):
        sig_marker = "***" if stats['highly_significant'] else ("**" if stats['significant'] else "ns")
        print(f"{comparison:<60} p={stats['p_value']:.4f} {sig_marker}")
    print("\nLegend: *** p<0.01, ** p<0.05, ns = not significant")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if baseline and public_cot:
        print(f"\n1. PUBLIC COT EFFECT:")
        print(f"   Baseline village win rate: {baseline['village_win_rate']:.1%}")
        print(f"   Public CoT village win rate: {public_cot['village_win_rate']:.1%}")
        print(f"   Change: {(public_cot['village_win_rate'] - baseline['village_win_rate'])*100:+.1f} percentage points")
    
    if baseline and probe_simple:
        print(f"\n2. PROBE (SIMPLE) EFFECT:")
        print(f"   Baseline village win rate: {baseline['village_win_rate']:.1%}")
        print(f"   Probe (Simple) village win rate: {probe_simple['village_win_rate']:.1%}")
        print(f"   Change: {(probe_simple['village_win_rate'] - baseline['village_win_rate'])*100:+.1f} percentage points")
    
    if baseline and probe_cautious:
        print(f"\n3. PROBE (CAUTIOUS) EFFECT:")
        print(f"   Baseline village win rate: {baseline['village_win_rate']:.1%}")
        print(f"   Probe (Cautious) village win rate: {probe_cautious['village_win_rate']:.1%}")
        print(f"   Change: {(probe_cautious['village_win_rate'] - baseline['village_win_rate'])*100:+.1f} percentage points")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
