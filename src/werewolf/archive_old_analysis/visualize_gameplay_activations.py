#!/usr/bin/env python3
"""
Visualize per-token gameplay activations for all players across all actions.

Shows each player's actions throughout the game with color-coded tokens based on
activation scores. Uses a yellow-to-red color scheme with black text for readability.

Includes full prompts from llm_log.txt to show context.

Usage:
    python visualize_gameplay_activations.py results/gameN/game_stats.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_llm_log(log_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse llm_log.txt to extract prompts for each player action.
    
    Returns:
        Dict mapping player names to list of prompt data:
        {
            'Player1': [
                {
                    'system_prompt': '...',
                    'user_prompt': '...',
                    'response': '...'
                },
                ...
            ]
        }
    """
    if not log_path.exists():
        print(f"Warning: llm_log.txt not found at {log_path}")
        return {}
    
    prompts = {}
    
    with open(log_path) as f:
        content = f.read()
    
    # Split by LLM Call sections
    sections = re.split(r'={80,}\n\[LLM Call for ([^\]]+)\]', content)
    
    # Skip the first empty section
    for i in range(1, len(sections), 2):
        player_name = sections[i].strip().split(']')[0]  # Extract just player name
        section_content = sections[i + 1] if i + 1 < len(sections) else ""
        
        # Extract system prompt
        system_match = re.search(r'--- SYSTEM PROMPT ---\n(.*?)\n\n--- USER PROMPT ---', 
                                section_content, re.DOTALL)
        system_prompt = system_match.group(1).strip() if system_match else ""
        
        # Extract user prompt
        user_match = re.search(r'--- USER PROMPT ---\n(.*?)\n\n--- LLM RESPONSE ---', 
                              section_content, re.DOTALL)
        user_prompt = user_match.group(1).strip() if user_match else ""
        
        # Extract response
        response_match = re.search(r'--- LLM RESPONSE ---\n(.*?)\n\n--- METADATA ---', 
                                  section_content, re.DOTALL)
        response = response_match.group(1).strip() if response_match else ""
        
        if player_name not in prompts:
            prompts[player_name] = []
        
        prompts[player_name].append({
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': response
        })
    
    return prompts


def score_to_color(score: float, min_score: float, max_score: float) -> str:
    """
    Convert activation score to background color.
    
    Uses yellow (low deception) -> orange -> red (high deception) gradient.
    Black text is readable against all these colors.
    
    Args:
        score: Activation score (typically negative, less negative = more deceptive)
        min_score: Minimum score in dataset (most negative)
        max_score: Maximum score in dataset (least negative/most deceptive)
    
    Returns:
        RGB color string
    """
    # Normalize score to 0-1 range (0 = min/most negative, 1 = max/least negative)
    if max_score == min_score:
        normalized = 0.5
    else:
        normalized = (score - min_score) / (max_score - min_score)
    
    # Yellow (safe) -> Orange -> Red (deceptive)
    # Yellow: rgb(255, 255, 0)
    # Orange: rgb(255, 165, 0)  
    # Red: rgb(255, 0, 0)
    
    if normalized < 0.5:
        # Yellow to Orange
        t = normalized * 2  # 0 to 1
        r = 255
        g = int(255 - (90 * t))  # 255 -> 165
        b = 0
    else:
        # Orange to Red
        t = (normalized - 0.5) * 2  # 0 to 1
        r = 255
        g = int(165 * (1 - t))  # 165 -> 0
        b = 0
    
    return f"rgb({r}, {g}, {b})"


def generate_html(game_stats: Dict, prompts: Dict[str, List[Dict]], output_path: Path) -> None:
    """Generate HTML visualization of gameplay activations."""
    
    # Extract player info and activations
    players = game_stats.get('players', [])
    player_activations = game_stats.get('player_activations', {})
    
    # Build player role lookup
    player_roles = {p['name']: p['role'] for p in players}
    
    # Find global min/max scores across all actions for consistent coloring
    all_scores = []
    for player_name, actions in player_activations.items():
        for action in actions:
            activations = action.get('activations', {})
            token_scores = activations.get('token_scores', [])
            all_scores.extend(token_scores)
    
    if not all_scores:
        print("No token scores found in game data")
        return
    
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    print(f"Score range: {min_score:.2f} to {max_score:.2f}")
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Werewolf Game {game_stats.get('game_id', 'N/A')} - Gameplay Activations</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .game-info {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .legend-gradient {{
            height: 30px;
            background: linear-gradient(to right, 
                rgb(255, 255, 0) 0%, 
                rgb(255, 165, 0) 50%, 
                rgb(255, 0, 0) 100%);
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
        }}
        
        .player-section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .player-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        
        .player-name {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }}
        
        .role-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: bold;
            color: white;
        }}
        
        .role-werewolf {{
            background-color: #dc3545;
        }}
        
        .role-villager {{
            background-color: #28a745;
        }}
        
        .role-seer {{
            background-color: #007bff;
        }}
        
        .action-entry {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 6px;
        }}
        
        .action-header {{
            font-weight: bold;
            color: #555;
            margin-bottom: 12px;
            font-size: 16px;
        }}
        
        .prompt-section {{
            background-color: #e9ecef;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .prompt-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 6px;
            font-family: 'Segoe UI', sans-serif;
        }}
        
        .prompt-content {{
            color: #212529;
            white-space: pre-wrap;
            margin-bottom: 12px;
        }}
        
        .action-text {{
            color: #666;
            margin-bottom: 10px;
            font-style: italic;
        }}
        
        .activation-score {{
            color: #888;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        
        .tokens-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            line-height: 2;
        }}
        
        .token {{
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: black;
            white-space: pre;
            cursor: help;
            transition: transform 0.1s;
        }}
        
        .token:hover {{
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <h1>Werewolf Game {game_stats.get('game_id', 'N/A')} - Per-Token Gameplay Activations</h1>
    <div class="game-info">
        Winner: {game_stats.get('winner', 'N/A')} | Total Turns: {game_stats.get('total_turns', 'N/A')}
    </div>
    
    <div class="legend">
        <div class="legend-title">Activation Score Legend</div>
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>Low Deception (Yellow: {min_score:.2f})</span>
            <span>Medium Deception (Orange)</span>
            <span>High Deception (Red: {max_score:.2f})</span>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            Hover over tokens to see exact scores. Less negative scores (closer to 0) indicate higher deception.
        </div>
    </div>
"""
    
    # Sort players: werewolves first, then others
    sorted_players = sorted(
        player_activations.items(),
        key=lambda x: (
            0 if player_roles.get(x[0]) == 'werewolf' else 1,
            x[0]
        )
    )
    
    # Generate player sections
    for player_name, actions in sorted_players:
        role = player_roles.get(player_name, 'unknown')
        role_class = f"role-{role}"
        
        player_prompts = prompts.get(player_name, [])
        
        html += f"""
    <div class="player-section">
        <div class="player-header">
            <span class="player-name">{player_name}</span>
            <span class="role-badge {role_class}">{role.upper()}</span>
        </div>
"""
        
        # Generate action entries
        for idx, action in enumerate(actions, 1):
            action_text = action.get('action', 'N/A')
            activations = action.get('activations', {})
            
            aggregate_score = activations.get('aggregate_score', 0.0)
            token_scores = activations.get('token_scores', [])
            tokens = activations.get('tokens', [])
            
            if not token_scores or not tokens:
                continue
            
            html += f"""
        <div class="action-entry">
            <div class="action-header">Action {idx}</div>
"""
            
            # Add prompt if available
            if idx <= len(player_prompts):
                prompt_data = player_prompts[idx - 1]
                system_prompt = prompt_data.get('system_prompt', '')
                user_prompt = prompt_data.get('user_prompt', '')
                
                # Escape HTML in prompts
                system_prompt_escaped = system_prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                user_prompt_escaped = user_prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                html += f"""
            <div class="prompt-section">
                <div class="prompt-label">System Prompt:</div>
                <div class="prompt-content">{system_prompt_escaped}</div>
                <div class="prompt-label">User Prompt:</div>
                <div class="prompt-content">{user_prompt_escaped}</div>
            </div>
"""
            
            html += f"""
            <div class="action-text">Generated: "{action_text}"</div>
            <div class="activation-score">Mean Score: {aggregate_score:.3f} | Tokens: {len(tokens)}</div>
            <div class="tokens-container">
"""
            
            # Generate colored tokens
            for token, score in zip(tokens, token_scores):
                color = score_to_color(score, min_score, max_score)
                # Escape HTML entities in token text
                token_escaped = (token
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;'))
                
                html += f'                <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
            
            html += """            </div>
        </div>
"""
        
        html += "    </div>\n"
    
    html += """
</body>
</html>
"""
    
    # Write HTML file
    output_path.write_text(html)
    print(f"Visualization saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_gameplay_activations.py <game_stats.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Load game stats
    with open(input_path) as f:
        game_stats = json.load(f)
    
    # Parse LLM log for prompts
    llm_log_path = input_path.parent / "llm_log.txt"
    prompts = parse_llm_log(llm_log_path)
    
    # Generate output filename
    output_path = input_path.parent / "gameplay_activations.html"
    
    # Generate HTML
    generate_html(game_stats, prompts, output_path)


if __name__ == "__main__":
    main()
