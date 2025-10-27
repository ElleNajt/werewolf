#!/usr/bin/env python3
"""
Generate HTML visualizations of per-token activation scores.

Usage:
    python visualize_activations.py <game_stats.json>
    
Creates role_reveal_viz.html showing token-level activations for all players.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def score_to_color(score: float, min_score: float, max_score: float) -> str:
    """
    Convert a score to a color using a diverging colormap.
    
    Uses a perceptually uniform diverging scheme:
    - Blue = low (negative) scores
    - White = middle scores  
    - Red = high (positive) scores
    """
    # Normalize score to [-1, 1]
    if max_score == min_score:
        normalized = 0.0
    else:
        normalized = 2 * (score - min_score) / (max_score - min_score) - 1
    
    # Clamp to [-1, 1]
    normalized = max(-1.0, min(1.0, normalized))
    
    # Map to color
    if normalized < 0:
        # Blue to white (negative scores)
        intensity = int(255 * (1 + normalized))
        return f"rgb({intensity}, {intensity}, 255)"
    else:
        # White to red (positive scores)
        intensity = int(255 * (1 - normalized))
        return f"rgb(255, {intensity}, {intensity})"


def create_token_html(token: str, score: float, min_score: float, max_score: float) -> str:
    """Create HTML for a single token with background color based on score."""
    color = score_to_color(score, min_score, max_score)
    
    # Escape HTML special characters
    token_display = (token.replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;')
                          .replace('\n', '↵'))
    
    # Replace spaces with visible character
    if token_display.strip() == '':
        token_display = '␣' * len(token)
    
    return f'<span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_display}</span>'


def visualize_role_reveal(game_stats: dict) -> str:
    """Generate HTML visualization of role-reveal activations."""
    
    html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Role-Reveal Activation Visualization</title>
    <style>
        body {
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #333;
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #555;
            margin-top: 30px;
            padding: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
        }
        
        .player-section {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .role-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            margin-left: 10px;
            font-size: 0.9em;
        }
        
        .werewolf { background-color: #ff4444; color: white; }
        .villager { background-color: #4CAF50; color: white; }
        .seer { background-color: #2196F3; color: white; }
        
        .tokens {
            line-height: 2.5;
            margin: 15px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        .token {
            display: inline-block;
            padding: 4px 6px;
            margin: 2px;
            border-radius: 3px;
            border: 1px solid #ccc;
            font-size: 13px;
            cursor: help;
            transition: transform 0.1s;
        }
        
        .token:hover {
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .stats {
            margin: 10px 0;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-size: 0.95em;
        }
        
        .legend {
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .legend-gradient {
            height: 30px;
            background: linear-gradient(to right, 
                rgb(0, 0, 255), 
                rgb(255, 255, 255), 
                rgb(255, 0, 0));
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Role-Reveal Activation Visualization</h1>
    <p>Visualization of probe activations when each player learns their role.</p>
    
    <div class="legend">
        <h3>Color Scale</h3>
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>Lower Scores (More Negative)</span>
            <span>Middle</span>
            <span>Higher Scores (Less Negative)</span>
        </div>
        <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
            Hover over tokens to see exact scores. Each token's background color represents its probe activation score.
        </p>
    </div>
''']
    
    # Get all role-reveal activations
    role_reveal = game_stats.get('role_reveal_activations', {})
    
    if not role_reveal:
        html_parts.append('<p>No role-reveal activations found in this game.</p>')
    else:
        # Get player roles
        player_roles = {p['name']: p['role'] for p in game_stats.get('players', [])}
        
        # Find global min/max scores for consistent coloring
        all_scores = []
        for player_data in role_reveal.values():
            if 'token_scores' in player_data:
                all_scores.extend(player_data['token_scores'])
        
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            
            html_parts.append(f'<p><strong>Score range across all players:</strong> {min_score:.3f} to {max_score:.3f}</p>')
            
            # Sort players by role (werewolves first for easy comparison)
            sorted_players = sorted(role_reveal.keys(), 
                                   key=lambda p: (player_roles.get(p, 'villager') != 'werewolf', p))
            
            for player_name in sorted_players:
                player_data = role_reveal[player_name]
                role = player_roles.get(player_name, 'unknown')
                
                # Skip if no token data
                if 'tokens' not in player_data or 'token_scores' not in player_data:
                    continue
                
                tokens = player_data['tokens']
                scores = player_data['token_scores']
                mean_score = player_data.get('prompt_mean_score', 0)
                
                html_parts.append(f'''
    <div class="player-section">
        <h2>{player_name}<span class="role-badge {role}">{role.upper()}</span></h2>
        <div class="stats">
            <strong>Mean Score:</strong> {mean_score:.4f} | 
            <strong>Tokens:</strong> {len(tokens)}
        </div>
        <div class="tokens">
''')
                
                # Add tokens with colors
                for token, score in zip(tokens, scores):
                    html_parts.append(create_token_html(token, score, min_score, max_score))
                
                html_parts.append('''
        </div>
    </div>
''')
        else:
            html_parts.append('<p>No token scores available.</p>')
    
    html_parts.append('''
</body>
</html>''')
    
    return '\n'.join(html_parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_activations.py <game_stats.json>")
        sys.exit(1)
    
    stats_file = Path(sys.argv[1])
    if not stats_file.exists():
        print(f"Error: {stats_file} not found")
        sys.exit(1)
    
    # Load game stats
    with open(stats_file) as f:
        game_stats = json.load(f)
    
    # Generate HTML
    html = visualize_role_reveal(game_stats)
    
    # Write to file
    output_file = stats_file.parent / "role_reveal_viz.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Visualization saved to: {output_file}")
    print(f"  Open in browser to view")


if __name__ == '__main__':
    main()
