#!/usr/bin/env python3
"""
Analyze probe activation clusters - separate werewolf vs villager statements.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_statements_with_activations(base_dir: Path):
    """Load all statements with their probe scores and roles."""
    statements = {
        'villager': [],
        'werewolf': []
    }
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return statements
    
    game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    games_found = 0
    for game_dir in game_dirs:
        stats_file = game_dir / "game_stats.json"
        if not stats_file.exists():
            continue
            
        with open(stats_file) as f:
            data = json.load(f)
        
        games_found += 1
        player_activations = data.get('player_activations', {})
        players = data.get('players', [])
        role_map = {p['name']: p['role'] for p in players}
        
        for player_name, player_acts in player_activations.items():
            role = role_map.get(player_name, 'unknown')
            if role not in ['villager', 'werewolf']:
                continue
            
            for act in player_acts:
                if 'activations' not in act:
                    continue
                
                activ = act['activations']
                action_text = act.get('action', '')
                reasoning = act.get('reasoning', '')
                
                # Get action score (what's shown to players)
                action_score = activ.get('action_score', None)
                
                if action_score is not None:
                    statements[role].append({
                        'score': action_score,
                        'action': action_text,
                        'reasoning': reasoning,
                        'player': player_name,
                        'game': game_dir.name
                    })
    
    print(f"Loaded {games_found} games")
    print(f"  Villager statements: {len(statements['villager'])}")
    print(f"  Werewolf statements: {len(statements['werewolf'])}")
    
    return statements

def identify_clusters(statements, role):
    """Identify potential clusters in the data."""
    scores = [s['score'] for s in statements]
    
    if not scores:
        return None
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    median = np.median(scores_array)
    
    # Define clusters based on distribution
    threshold_low = mean - 0.5 * std
    threshold_high = mean + 0.5 * std
    
    low_cluster = [s for s in statements if s['score'] < threshold_low]
    mid_cluster = [s for s in statements if threshold_low <= s['score'] <= threshold_high]
    high_cluster = [s for s in statements if s['score'] > threshold_high]
    
    print(f"\n{role.upper()} CLUSTERS:")
    print(f"  Low (< {threshold_low:.2f}): {len(low_cluster)} statements")
    print(f"  Mid ({threshold_low:.2f} to {threshold_high:.2f}): {len(mid_cluster)} statements")
    print(f"  High (> {threshold_high:.2f}): {len(high_cluster)} statements")
    
    return {
        'low': low_cluster,
        'mid': mid_cluster,
        'high': high_cluster,
        'thresholds': (threshold_low, threshold_high)
    }

def plot_clusters(villager_statements, werewolf_statements, output_file: str):
    """Plot histogram showing clusters for both roles."""
    
    v_scores = [s['score'] for s in villager_statements]
    w_scores = [s['score'] for s in werewolf_statements]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Combined histogram
    all_scores = v_scores + w_scores
    bins = np.linspace(min(all_scores) - 0.5, max(all_scores) + 0.5, 50)
    
    ax1.hist(v_scores, bins=bins, alpha=0.6, label='Villagers', 
             color='#2ecc71', density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(w_scores, bins=bins, alpha=0.6, label='Werewolves',
             color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
    
    v_mean = np.mean(v_scores)
    w_mean = np.mean(w_scores)
    ax1.axvline(v_mean, color='#27ae60', linestyle='--', linewidth=2, 
                label=f'Villager mean: {v_mean:.2f}')
    ax1.axvline(w_mean, color='#c0392b', linestyle='--', linewidth=2,
                label=f'Werewolf mean: {w_mean:.2f}')
    
    ax1.set_xlabel('Action Score (more negative = more suspicious)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('70B Probe: Action Score Distribution (Villagers vs Werewolves)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Separate histograms
    ax2_top = ax2
    ax2_bottom = ax2.twinx()
    
    ax2_top.hist(v_scores, bins=bins, alpha=0.7, label='Villagers', 
                 color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax2_bottom.hist(w_scores, bins=bins, alpha=0.7, label='Werewolves',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax2_top.set_xlabel('Action Score (more negative = more suspicious)', fontsize=12)
    ax2_top.set_ylabel('Villager Count', fontsize=12, color='#27ae60')
    ax2_bottom.set_ylabel('Werewolf Count', fontsize=12, color='#c0392b')
    ax2_top.set_title('70B Probe: Separate Distributions', fontsize=14, fontweight='bold')
    
    ax2_top.tick_params(axis='y', labelcolor='#27ae60')
    ax2_bottom.tick_params(axis='y', labelcolor='#c0392b')
    
    ax2_top.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCluster plot saved to: {output_file}")
    plt.close()

def save_example_statements(clusters, role, output_file: str, n_examples=10):
    """Save example statements from each cluster to a file."""
    
    with open(output_file, 'w') as f:
        f.write(f"# {role.upper()} STATEMENT EXAMPLES BY CLUSTER\n")
        f.write(f"# Analysis of 70B Probe Action Scores\n\n")
        
        for cluster_name in ['low', 'mid', 'high']:
            cluster = clusters[cluster_name]
            if not cluster:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{cluster_name.upper()} CLUSTER ({len(cluster)} statements)\n")
            f.write(f"{'='*80}\n\n")
            
            # Sort by score and take examples
            sorted_cluster = sorted(cluster, key=lambda x: x['score'])
            examples = sorted_cluster[:n_examples] if cluster_name == 'low' else \
                      sorted_cluster[len(sorted_cluster)//2 - n_examples//2:len(sorted_cluster)//2 + n_examples//2] if cluster_name == 'mid' else \
                      sorted_cluster[-n_examples:]
            
            for i, stmt in enumerate(examples, 1):
                f.write(f"{i}. Score: {stmt['score']:.2f}\n")
                f.write(f"   Game: {stmt['game']}, Player: {stmt['player']}\n")
                if stmt['reasoning']:
                    f.write(f"   Reasoning: {stmt['reasoning'][:200]}...\n" if len(stmt['reasoning']) > 200 else f"   Reasoning: {stmt['reasoning']}\n")
                f.write(f"   Action: {stmt['action'][:200]}...\n" if len(stmt['action']) > 200 else f"   Action: {stmt['action']}\n")
                f.write(f"\n")
    
    print(f"Example statements saved to: {output_file}")

def main():
    print("\n" + "="*80)
    print("70B PROBE CLUSTER ANALYSIS")
    print("="*80)
    
    probe_dir = Path("../../results/70b_probe_villagers_9622b6b_2025-10-30_17-57-30")
    
    print(f"\nLoading statements from:\n  {probe_dir}")
    statements = load_statements_with_activations(probe_dir)
    
    # Identify clusters
    v_clusters = identify_clusters(statements['villager'], 'villager')
    w_clusters = identify_clusters(statements['werewolf'], 'werewolf')
    
    # Plot clusters
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    plot_clusters(statements['villager'], statements['werewolf'], 
                 "../../results/70b_probe_clusters.png")
    
    # Save example statements
    save_example_statements(v_clusters, 'villager', 
                           "../../results/70b_villager_statement_examples.txt", n_examples=15)
    save_example_statements(w_clusters, 'werewolf', 
                           "../../results/70b_werewolf_statement_examples.txt", n_examples=15)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    v_scores = [s['score'] for s in statements['villager']]
    w_scores = [s['score'] for s in statements['werewolf']]
    
    print(f"\nVILLAGERS (n={len(v_scores)}):")
    print(f"  Mean: {np.mean(v_scores):.3f}")
    print(f"  Std: {np.std(v_scores):.3f}")
    print(f"  Median: {np.median(v_scores):.3f}")
    print(f"  Range: [{np.min(v_scores):.2f}, {np.max(v_scores):.2f}]")
    
    print(f"\nWEREWOLVES (n={len(w_scores)}):")
    print(f"  Mean: {np.mean(w_scores):.3f}")
    print(f"  Std: {np.std(w_scores):.3f}")
    print(f"  Median: {np.median(w_scores):.3f}")
    print(f"  Range: [{np.min(w_scores):.2f}, {np.max(w_scores):.2f}]")
    
    print(f"\nDIFFERENCE (W - V):")
    print(f"  Mean difference: {np.mean(w_scores) - np.mean(v_scores):.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/70b_probe_clusters.png")
    print("  - results/70b_villager_statement_examples.txt")
    print("  - results/70b_werewolf_statement_examples.txt")
    print("="*80)

if __name__ == "__main__":
    main()
