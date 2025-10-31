#!/usr/bin/env python3
"""
Extract complete 8B werewolf and villager statements with probe scores to CSV.
No truncation - full reasoning and action text.
"""

import json
import re
import csv
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
    """Load all statements from all games in directory."""
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


def write_statements_to_csv(statements: List[Dict[str, Any]], output_path: Path):
    """Write complete statements with scores to CSV."""
    
    # Sort by action score (low to high)
    sorted_statements = sorted(statements, key=lambda x: x['action_score'])
    
    fieldnames = [
        'game',
        'player', 
        'role',
        'prompt_score',
        'cot_score',
        'action_score',
        'cot_num_tokens',
        'action_num_tokens',
        'reasoning',
        'action'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_statements)


def main():
    base_dir = Path("../../results/final_8b_probe_simple_d79edc0_8288738")
    
    print(f"\nExtracting complete 8B statements from: {base_dir}")
    print("=" * 80)
    
    statements = load_all_statements(base_dir)
    
    print(f"\nFound:")
    print(f"  Werewolf statements: {len(statements['werewolf'])}")
    print(f"  Villager statements: {len(statements['villager'])}")
    
    # Write to CSV files
    output_dir = Path("../../results")
    
    werewolf_file = output_dir / "8b_werewolf_complete_statements.csv"
    villager_file = output_dir / "8b_villager_complete_statements.csv"
    
    print(f"\nWriting complete statements to CSV files...")
    write_statements_to_csv(statements['werewolf'], werewolf_file)
    write_statements_to_csv(statements['villager'], villager_file)
    
    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nGenerated files:")
    print(f"  - {werewolf_file}")
    print(f"  - {villager_file}")
    print()


if __name__ == "__main__":
    main()
