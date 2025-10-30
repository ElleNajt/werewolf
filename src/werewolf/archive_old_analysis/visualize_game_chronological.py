#!/usr/bin/env python3
"""
Visualize per-token gameplay activations in chronological game order.

Shows the full game flow turn-by-turn with prompts and color-coded token activations.
Uses yellow-to-red color scheme with black text for readability.

Usage:
    python visualize_game_chronological.py results/gameN/game_stats.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_llm_log_chronological(log_path: Path) -> List[Dict]:
    """
    Parse llm_log.txt to extract ALL LLM calls in chronological order.
    
    Returns:
        List of dicts in order:
        [
            {
                'player': 'Player2',
                'role': 'werewolf',
                'system_prompt': '...',
                'user_prompt': '...',
                'response': '...'
            },
            ...
        ]
    """
    if not log_path.exists():
        print(f"Warning: llm_log.txt not found at {log_path}")
        return []
    
    actions = []
    
    with open(log_path) as f:
        content = f.read()
    
    # Split by LLM Call sections
    sections = re.split(r'={80,}\n\[LLM Call for ([^\]]+)\]', content)
    
    # Skip the first empty section
    for i in range(1, len(sections), 2):
        player_info = sections[i].strip()
        # Extract player name and role from "[LLM Call for Player2] (werewolf)"
        player_match = re.match(r'([^\s]+)', player_info)
        role_match = re.search(r'\(([^\)]+)\)', player_info)
        
        player_name = player_match.group(1) if player_match else "Unknown"
        role = role_match.group(1) if role_match else "unknown"
        
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
        
        # Extract metadata
        metadata_match = re.search(r'--- METADATA ---\n(.*?)(?=\n={80,}|\Z)', 
                                  section_content, re.DOTALL)
        metadata_str = metadata_match.group(1).strip() if metadata_match else "{}"
        
        try:
            metadata = json.loads(metadata_str)
        except:
            metadata = {}
        
        actions.append({
            'player': player_name,
            'role': role,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': response,
            'metadata': metadata
        })
    
    return actions


def score_to_color(score: float, min_score: float, max_score: float) -> str:
    """Convert activation score to background color (yellow -> orange -> red)."""
    if max_score == min_score:
        normalized = 0.5
    else:
        normalized = (score - min_score) / (max_score - min_score)
    
    # Clamp normalized value to 0-1 range to handle outliers
    normalized = max(0.0, min(1.0, normalized))
    
    if normalized < 0.5:
        t = normalized * 2
        r = 255
        g = int(255 - (90 * t))
        b = 0
    else:
        t = (normalized - 0.5) * 2
        r = 255
        g = int(165 * (1 - t))
        b = 0
    
    return f"rgb({r}, {g}, {b})"


def generate_html(game_stats: Dict, actions: List[Dict], output_path: Path) -> None:
    """Generate HTML visualization in chronological order."""
    
    # Build player -> role mapping from game_stats
    player_roles = {}
    for player in game_stats.get('players', []):
        player_roles[player['name']] = player['role']
    
    # Update action roles from game_stats if they're missing/unknown
    for action in actions:
        player_name = action['player']
        if action['role'] == 'unknown' and player_name in player_roles:
            action['role'] = player_roles[player_name]
    
    # Find global min/max scores across ALL token types
    all_scores = []
    for action in actions:
        metadata = action.get('metadata', {})
        activations = metadata.get('activations', {})
        
        # Collect scores from all sources
        all_scores.extend(activations.get('token_scores', []))
        all_scores.extend(activations.get('cot_prompt_token_scores', []))
        all_scores.extend(activations.get('cot_generation_token_scores', []))
        all_scores.extend(activations.get('action_prompt_token_scores', []))
        all_scores.extend(activations.get('action_generation_token_scores', []))
    
    if not all_scores:
        print("No token scores found in game data")
        return
    
    # Use percentiles to handle outliers
    import numpy as np
    all_scores_array = np.array(all_scores)
    min_score = np.percentile(all_scores_array, 1)  # 1st percentile
    max_score = np.percentile(all_scores_array, 99)  # 99th percentile
    
    print(f"Absolute range: {all_scores_array.min():.2f} to {all_scores_array.max():.2f}")
    print(f"Percentile range (1-99): {min_score:.2f} to {max_score:.2f}")
    
    print(f"Score range: {min_score:.2f} to {max_score:.2f}")
    print(f"Total actions: {len(actions)}")
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Werewolf Game {game_stats.get('game_id', 'N/A')} - Chronological View</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
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
        
        .action-card {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #ddd;
        }}
        
        .action-card.werewolf {{
            border-left-color: #dc3545;
            background-color: #ffe6e6;  /* Light red background */
        }}
        
        .action-card.villager {{
            border-left-color: #28a745;
            background-color: #e6f4ea;  /* Light green background */
        }}
        
        .action-card.seer {{
            border-left-color: #007bff;
            background-color: #e6f2ff;  /* Light blue background */
        }}
        
        .action-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .player-name {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        
        .role-badge {{
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        
        .role-badge.werewolf {{
            background-color: #dc3545;
        }}
        
        .role-badge.villager {{
            background-color: #28a745;
        }}
        
        .role-badge.seer {{
            background-color: #007bff;
        }}
        
        .action-number {{
            color: #999;
            font-size: 14px;
            margin-left: auto;
        }}
        
        .prompt-section {{
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #e9ecef;
        }}
        
        .prompt-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 6px;
            margin-top: 8px;
            font-family: 'Segoe UI', sans-serif;
        }}
        
        .prompt-label:first-child {{
            margin-top: 0;
        }}
        
        .prompt-content {{
            color: #212529;
            white-space: pre-wrap;
            margin-bottom: 8px;
        }}
        
        .generated-section {{
            margin-bottom: 12px;
        }}
        
        .generated-label {{
            font-weight: bold;
            color: #555;
            margin-bottom: 6px;
        }}
        
        .activation-score {{
            color: #888;
            font-size: 13px;
            margin-bottom: 10px;
        }}
        
        .tokens-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            line-height: 2.2;
        }}
        
        .token {{
            padding: 3px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: black;
            white-space: pre;
            cursor: help;
            transition: all 0.1s;
            position: relative;
        }}
        
        .token:hover {{
            transform: scale(1.15);
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        
        .token:hover::after {{
            content: attr(title);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            margin-bottom: 5px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h1>Werewolf Game {game_stats.get('game_id', 'N/A')} - Chronological View</h1>
    <div class="game-info">
        Winner: {game_stats.get('winner', 'N/A')} | Total Turns: {game_stats.get('total_turns', 'N/A')} | Total Actions: {len(actions)}
    </div>
    
    <div class="legend">
        <div class="legend-title">Activation Score Legend</div>
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>Low Deception (Yellow: {min_score:.2f})</span>
            <span>Medium Deception (Orange)</span>
            <span>High Deception (Red: {max_score:.2f})</span>
        </div>
        <div style="margin-top: 15px; font-size: 13px; color: #333;">
            <strong>Four Activation Phases Per Action:</strong>
        </div>
        <div style="margin-top: 8px; font-size: 12px; color: #666; line-height: 1.6;">
            <div><strong>CoT Prompt:</strong> Activations when reading "provide your private reasoning"</div>
            <div><strong>CoT Generation:</strong> Activations when generating internal reasoning (private)</div>
            <div><strong>Action Prompt:</strong> Activations when reading the full game context + CoT</div>
            <div><strong>Action Generation:</strong> Activations when generating the public statement/action</div>
        </div>
        <div style="margin-top: 10px; font-size: 11px; color: #888;">
            Hover over any colored token to see its exact score. Less negative scores = more deceptive.
        </div>
    </div>
"""
    
    # Generate action cards in chronological order
    for idx, action in enumerate(actions, 1):
        player_name = action['player']
        role = action['role']
        system_prompt = action['system_prompt']
        user_prompt = action['user_prompt']
        response = action['response']
        metadata = action.get('metadata', {})
        
        activations = metadata.get('activations', {})
        aggregate_score = activations.get('aggregate_score', 0.0)
        token_scores = activations.get('token_scores', [])
        tokens = activations.get('tokens', [])
        
        # Escape HTML
        system_prompt_escaped = system_prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        user_prompt_escaped = user_prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        response_escaped = response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        html += f"""
    <div class="action-card {role}">
        <div class="action-header">
            <span class="player-name">{player_name}</span>
            <span class="role-badge {role}">{role.upper()}</span>
            <span class="action-number">#{idx}</span>
        </div>
        
        <div class="prompt-section">
            <div class="prompt-label">System Prompt:</div>
            <div class="prompt-content">{system_prompt_escaped}</div>
            <div class="prompt-label">User Prompt:</div>
            <div class="prompt-content">{user_prompt_escaped}</div>
        </div>
        
        <div class="generated-section">
            <div class="generated-label">Generated Response:</div>
            <div class="prompt-content" style="background: #e9ecef; padding: 8px; border-radius: 4px;">{response_escaped}</div>
        </div>
"""
        
        # Display activation scores
        cot_prompt_mean = activations.get('cot_prompt_mean_score')
        cot_gen_mean = activations.get('cot_generation_mean_score')
        action_prompt_mean = activations.get('action_prompt_mean_score')
        action_gen_mean = activations.get('action_generation_mean_score')
        
        if any([cot_prompt_mean, cot_gen_mean, action_prompt_mean, action_gen_mean]):
            html += '<div class="activation-score"><strong>Mean Scores:</strong> '
            if cot_prompt_mean is not None:
                html += f'CoT Prompt: {cot_prompt_mean:.3f} | '
            if cot_gen_mean is not None:
                html += f'CoT Generation: {cot_gen_mean:.3f} | '
            if action_prompt_mean is not None:
                html += f'Action Prompt: {action_prompt_mean:.3f} | '
            if action_gen_mean is not None:
                html += f'Action Generation: {action_gen_mean:.3f}'
            html += '</div>\n'
            html += '<div style="font-size: 11px; color: #888; margin-top: 4px;">Hover over colored tokens to see individual scores. Less negative = more deceptive.</div>\n'
        
        # Display CoT prompt tokens if available
        cot_prompt_tokens = activations.get('cot_prompt_tokens', [])
        cot_prompt_scores = activations.get('cot_prompt_token_scores', [])
        if cot_prompt_tokens and cot_prompt_scores:
            html += '<div style="margin-top: 10px;"><strong>CoT Prompt Tokens:</strong></div>\n'
            html += '<div class="tokens-container">\n'
            for token, score in zip(cot_prompt_tokens, cot_prompt_scores):
                color = score_to_color(score, min_score, max_score)
                token_escaped = (token.replace('&', '&amp;').replace('<', '&lt;')
                                .replace('>', '&gt;').replace('"', '&quot;'))
                html += f'    <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
            html += '</div>\n'
        
        # Display CoT generation tokens if available
        cot_gen_tokens = activations.get('cot_generation_tokens', [])
        cot_gen_scores = activations.get('cot_generation_token_scores', [])
        if cot_gen_tokens and cot_gen_scores:
            html += '<div style="margin-top: 10px;"><strong>CoT Generation Tokens:</strong></div>\n'
            html += '<div class="tokens-container">\n'
            for token, score in zip(cot_gen_tokens, cot_gen_scores):
                color = score_to_color(score, min_score, max_score)
                token_escaped = (token.replace('&', '&amp;').replace('<', '&lt;')
                                .replace('>', '&gt;').replace('"', '&quot;'))
                html += f'    <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
            html += '</div>\n'
        
        # Display action prompt tokens if available
        action_prompt_tokens = activations.get('action_prompt_tokens', [])
        action_prompt_scores = activations.get('action_prompt_token_scores', [])
        if action_prompt_tokens and action_prompt_scores:
            html += '<div style="margin-top: 10px;"><strong>Action Prompt Tokens:</strong></div>\n'
            html += '<div class="tokens-container">\n'
            for token, score in zip(action_prompt_tokens, action_prompt_scores):
                color = score_to_color(score, min_score, max_score)
                token_escaped = (token.replace('&', '&amp;').replace('<', '&lt;')
                                .replace('>', '&gt;').replace('"', '&quot;'))
                html += f'    <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
            html += '</div>\n'
        
        # Display action generation tokens if available
        action_gen_tokens = activations.get('action_generation_tokens', [])
        action_gen_scores = activations.get('action_generation_token_scores', [])
        if action_gen_tokens and action_gen_scores:
            html += '<div style="margin-top: 10px;"><strong>Action Generation Tokens:</strong></div>\n'
            html += '<div class="tokens-container">\n'
            for token, score in zip(action_gen_tokens, action_gen_scores):
                color = score_to_color(score, min_score, max_score)
                token_escaped = (token.replace('&', '&amp;').replace('<', '&lt;')
                                .replace('>', '&gt;').replace('"', '&quot;'))
                html += f'    <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
            html += '</div>\n'
        
        # Legacy: also show old token_scores if no new format
        if not any([cot_prompt_tokens, cot_gen_tokens, action_prompt_tokens, action_gen_tokens]):
            if token_scores and tokens:
                html += f'<div class="activation-score">Generation Score: {aggregate_score:.3f} | Tokens: {len(tokens)}</div>\n'
                html += '<div class="tokens-container">\n'
                for token, score in zip(tokens, token_scores):
                    color = score_to_color(score, min_score, max_score)
                    token_escaped = (token.replace('&', '&amp;').replace('<', '&lt;')
                                    .replace('>', '&gt;').replace('"', '&quot;'))
                    html += f'    <span class="token" style="background-color: {color};" title="Score: {score:.3f}">{token_escaped}</span>\n'
                html += '</div>\n'
            else:
                html += '<div style="color: #999; font-style: italic;">No token activations available</div>\n'
        
        html += "    </div>\n"
    
    html += """
</body>
</html>
"""
    
    output_path.write_text(html)
    print(f"Chronological visualization saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_game_chronological.py <game_stats.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Load game stats
    with open(input_path) as f:
        game_stats = json.load(f)
    
    # Parse LLM log for chronological actions
    llm_log_path = input_path.parent / "llm_log.txt"
    actions = parse_llm_log_chronological(llm_log_path)
    
    # Generate output filename
    output_path = input_path.parent / "game_chronological.html"
    
    # Generate HTML
    generate_html(game_stats, actions, output_path)


if __name__ == "__main__":
    main()
