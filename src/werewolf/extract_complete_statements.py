#!/usr/bin/env python3
"""
Extract complete werewolf and villager statements with probe scores.
No truncation - full reasoning and action text.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any


def extract_statements_from_llm_log(llm_log_path: Path, game_name: str) -> List[Dict[str, Any]]:
    """Parse LLM log to extract complete statements with scores."""
    statements = []
    
    with open(llm_log_path, 'r') as f:
        content = f.read()
    
    # Use regex to find all LLM call blocks
    pattern = r'={80}\n\[LLM Call for (Player\d+)\] \((werewolf|villager)\)\n={80}(.*?)(?=\n={80}\n\[LLM Call|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        player = match.group(1)
        role = match.group(2)
        block_content = match.group(3)
        
        # Extract LLM response JSON
        response_match = re.search(r'--- LLM RESPONSE ---\s+({[^}]+})', block_content)
        if not response_match:
            continue
        
        try:
            response_json = json.loads(response_match.group(1))
        except json.JSONDecodeError:
            continue
        
        # Extract metadata JSON
        metadata_match = re.search(r'--- METADATA ---\s+(\{[\s\S]+?\n\})\n', block_content)
        if not metadata_match:
            continue
        
        try:
            metadata = json.loads(metadata_match.group(1))
        except json.JSONDecodeError:
            continue
        
        activations = metadata.get('activations', {})
        
        # Only include statements with action scores (day statements, not night votes)
        if activations.get('action_score') is None:
            continue
        
        statement = {
            'game': game_name,
            'player': player,
            'role': role,
            'reasoning': response_json.get('reasoning', ''),
            'action': response_json.get('action', ''),
            'prompt_score': activations.get('prompt_score'),
            'cot_score': activations.get('cot_score'),
            'action_score': activations.get('action_score'),
            'cot_num_tokens': activations.get('cot_num_tokens'),
            'action_num_tokens': activations.get('action_num_tokens'),
        }
        
        statements.append(statement)
    
    return statements


def load_all_statements(base_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all statements from all games."""
    statements = {'werewolf': [], 'villager': []}
    
    game_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    print(f"Loading statements from {len(game_dirs)} games...")
    
    for game_dir in game_dirs:
        llm_log = game_dir / 'llm_log.txt'
        if not llm_log.exists():
            continue
        
        game_statements = extract_statements_from_llm_log(llm_log, game_dir.name)
        
        for stmt in game_statements:
            statements[stmt['role']].append(stmt)
        
        print(f"  {game_dir.name}: {len(game_statements)} statements")
    
    return statements


def write_statements_to_file(statements: List[Dict[str, Any]], output_path: Path, role: str):
    """Write complete statements with scores to file."""
    
    # Sort by action score (low to high)
    sorted_statements = sorted(statements, key=lambda x: x['action_score'])
    
    with open(output_path, 'w') as f:
        f.write(f"# COMPLETE {role.upper()} STATEMENTS WITH PROBE SCORES\n")
        f.write(f"# Total statements: {len(statements)}\n")
        f.write(f"# Sorted by action score (low to high = more to less suspicious)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, stmt in enumerate(sorted_statements, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"STATEMENT {i}/{len(statements)}\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write(f"Game: {stmt['game']}\n")
            f.write(f"Player: {stmt['player']}\n")
            f.write(f"Role: {stmt['role']}\n\n")
            
            f.write(f"PROBE SCORES:\n")
            f.write(f"  Prompt Score:  {stmt['prompt_score']:.2f}\n")
            f.write(f"  CoT Score:     {stmt['cot_score']:.2f} ({stmt['cot_num_tokens']} tokens)\n")
            f.write(f"  Action Score:  {stmt['action_score']:.2f} ({stmt['action_num_tokens']} tokens)\n\n")
            
            f.write(f"REASONING (private thoughts):\n")
            f.write("-" * 80 + "\n")
            f.write(stmt['reasoning'])
            f.write("\n" + "-" * 80 + "\n\n")
            
            f.write(f"ACTION (public statement):\n")
            f.write("-" * 80 + "\n")
            f.write(stmt['action'])
            f.write("\n" + "-" * 80 + "\n")


def main():
    base_dir = Path("../../results/70b_probe_villagers_9622b6b_2025-10-30_17-57-30")
    
    print(f"\nExtracting complete statements from: {base_dir}")
    print("=" * 80)
    
    statements = load_all_statements(base_dir)
    
    print(f"\nFound:")
    print(f"  Werewolf statements: {len(statements['werewolf'])}")
    print(f"  Villager statements: {len(statements['villager'])}")
    
    # Write to files
    output_dir = Path("../../results")
    
    werewolf_file = output_dir / "70b_werewolf_complete_statements.txt"
    villager_file = output_dir / "70b_villager_complete_statements.txt"
    
    print(f"\nWriting complete statements to files...")
    write_statements_to_file(statements['werewolf'], werewolf_file, 'werewolf')
    write_statements_to_file(statements['villager'], villager_file, 'villager')
    
    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nGenerated files:")
    print(f"  - {werewolf_file}")
    print(f"  - {villager_file}")
    print()


if __name__ == "__main__":
    main()
