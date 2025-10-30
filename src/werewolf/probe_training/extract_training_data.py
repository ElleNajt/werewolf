"""
Extract training data for trust/collusion probes from werewolf game logs.

This script processes game statistics and creates training datasets for probes that detect:
1. Trust/alliance between players (same team = 1, different teams = 0)
2. Collusion detection (werewolves = 1, others = 0)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys

# Add deception-detection to path
DECEPTION_DETECTION_PATH = Path(__file__).parent.parent.parent.parent / "deception-detection"
sys.path.insert(0, str(DECEPTION_DETECTION_PATH))

from deception_detection.data.base import DialogueDataset, Message
from deception_detection.types import Label


@dataclass
class PlayerInteraction:
    """A single player's action/statement with context."""
    game_id: int
    turn: int
    player: str
    player_role: str
    action_type: str  # "statement", "vote", "night_kill"
    action: str
    target: Optional[str] = None  # Who they're talking about or voting for
    target_role: Optional[str] = None
    same_team: Optional[bool] = None  # True if player and target on same team
    activations: Optional[Dict] = None  # Activation data if available


def load_game_data(game_dir: Path) -> Tuple[Dict, Dict]:
    """Load game stats and extract player roles."""
    stats_file = game_dir / "game_stats.json"
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Build role mapping
    roles = {}
    for player_data in stats['players']:
        roles[player_data['name']] = player_data['role']
    
    return stats, roles


def are_same_team(role1: str, role2: str) -> bool:
    """Check if two roles are on the same team."""
    werewolf_team = {'werewolf'}
    village_team = {'villager', 'seer'}
    
    role1_lower = role1.lower()
    role2_lower = role2.lower()
    
    if role1_lower in werewolf_team and role2_lower in werewolf_team:
        return True
    if role1_lower in village_team and role2_lower in village_team:
        return True
    return False


def extract_interactions(game_dir: Path) -> List[PlayerInteraction]:
    """Extract all player interactions from a game with team labels."""
    stats, roles = load_game_data(game_dir)
    game_id = stats['game_id']
    interactions = []
    
    # Extract activations if available
    activations_by_player = stats.get('player_activations', {})
    
    # Process history
    for event in stats['history']:
        turn = event['turn']
        event_type = event['type']
        data = event['data']
        
        if event_type == 'day_statement':
            # Extract statement
            player = data['player']
            statement = data['statement']
            player_role = roles[player]
            
            # Get activations if available
            player_acts = activations_by_player.get(player, [])
            act_data = None
            for act in player_acts:
                if act.get('action') == statement:
                    act_data = act.get('activations')
                    break
            
            # Create interaction (no specific target for general statements)
            interactions.append(PlayerInteraction(
                game_id=game_id,
                turn=turn,
                player=player,
                player_role=player_role,
                action_type='statement',
                action=statement,
                activations=act_data
            ))
        
        elif event_type == 'day_elimination':
            # Extract votes - these have explicit targets
            voter_data = data.get('votes', {})
            victim = data['victim']
            
            # Reconstruct individual votes
            for player in roles.keys():
                if player in stats['players'] and any(p['name'] == player and not p.get('eliminated_turn', 100) < turn for p in stats['players']):
                    # Find who this player voted for
                    player_role = roles[player]
                    
                    # Get activations for vote
                    player_acts = activations_by_player.get(player, [])
                    act_data = None
                    # Look for vote action in activations
                    for act in player_acts:
                        if act.get('action') in roles.keys():  # Vote actions are player names
                            act_data = act.get('activations')
                            voted_for = act.get('action')
                            voted_for_role = roles.get(voted_for)
                            
                            if voted_for_role:
                                same_team = are_same_team(player_role, voted_for_role)
                                
                                interactions.append(PlayerInteraction(
                                    game_id=game_id,
                                    turn=turn,
                                    player=player,
                                    player_role=player_role,
                                    action_type='vote',
                                    action=voted_for,
                                    target=voted_for,
                                    target_role=voted_for_role,
                                    same_team=same_team,
                                    activations=act_data
                                ))
        
        elif event_type == 'night_kill':
            # Werewolf votes for night kill
            werewolf_votes = data.get('werewolf_votes', {})
            for werewolf, target in werewolf_votes.items():
                werewolf_role = roles[werewolf]
                target_role = roles.get(target)
                
                if target_role:
                    same_team = are_same_team(werewolf_role, target_role)
                    
                    # Get activations
                    player_acts = activations_by_player.get(werewolf, [])
                    act_data = None
                    for act in player_acts:
                        if act.get('action') == target:
                            act_data = act.get('activations')
                            break
                    
                    interactions.append(PlayerInteraction(
                        game_id=game_id,
                        turn=turn,
                        player=werewolf,
                        player_role=werewolf_role,
                        action_type='night_kill',
                        action=target,
                        target=target,
                        target_role=target_role,
                        same_team=same_team,
                        activations=act_data
                    ))
    
    return interactions


def extract_all_games(results_dir: Path) -> List[PlayerInteraction]:
    """Extract interactions from all games in a results directory."""
    all_interactions = []
    
    game_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    
    print(f"Processing {len(game_dirs)} games from {results_dir}...")
    
    for game_dir in game_dirs:
        try:
            interactions = extract_interactions(game_dir)
            all_interactions.extend(interactions)
            print(f"  {game_dir.name}: {len(interactions)} interactions")
        except Exception as e:
            print(f"  {game_dir.name}: ERROR - {e}")
    
    return all_interactions


def create_trust_probe_dataset(interactions: List[PlayerInteraction]) -> Tuple[List, List]:
    """
    Create dataset for trust probe training.
    
    ONLY uses werewolf interactions because only werewolves have ground truth
    about who their allies are. Villagers don't know team alignments.
    
    For werewolf interactions with a target:
    - Label = 1 if target is another werewolf (trust - they know ally)
    - Label = 0 if target is a villager (distrust - they know enemy)
    
    Returns:
        dialogues: List of dialogue messages
        labels: List of labels (1 = trust ally, 0 = distrust enemy)
    """
    dialogues = []
    labels = []
    
    for interaction in interactions:
        # Only use interactions with a clear target and team label
        if interaction.target is None or interaction.same_team is None:
            continue
        
        # CRITICAL: Only use werewolf interactions
        # Villagers don't have ground truth for trust (they don't know teams)
        if interaction.player_role.lower() != 'werewolf':
            continue
        
        # Create prompt that asks about the target
        system_msg = Message(
            role="system",
            content=f"You are {interaction.player}, a {interaction.player_role} in a Werewolf game.",
            is_assistant_response=False
        )
        
        # User asks about the target
        user_msg = Message(
            role="user",
            content=f"How do you feel about {interaction.target}?",
            is_assistant_response=False
        )
        
        # Assistant responds with the action
        assistant_msg = Message(
            role="assistant",
            content=interaction.action,
            is_assistant_response=True
        )
        
        dialogue = [system_msg, user_msg, assistant_msg]
        dialogues.append(dialogue)
        
        # Label: 1 = same team (trust - knows ally), 0 = different team (distrust - knows enemy)
        label = Label.HONEST if interaction.same_team else Label.DECEPTIVE
        labels.append(label)
    
    return dialogues, labels


def create_collusion_probe_dataset(interactions: List[PlayerInteraction]) -> Tuple[List, List]:
    """
    Create dataset for collusion probe training.
    
    - Label = 1 if player is werewolf (colluding team)
    - Label = 0 if player is villager/seer (non-colluding)
    
    Returns:
        dialogues: List of dialogue messages
        labels: List of labels (1 = werewolf, 0 = villager/seer)
    """
    dialogues = []
    labels = []
    
    for interaction in interactions:
        # Use all interactions (statements, votes, etc.)
        system_msg = Message(
            role="system",
            content=f"You are {interaction.player} in a Werewolf game.",
            is_assistant_response=False
        )
        
        # Generic prompt
        user_msg = Message(
            role="user",
            content="What do you say?",
            is_assistant_response=False
        )
        
        # Assistant responds with the action
        assistant_msg = Message(
            role="assistant",
            content=interaction.action,
            is_assistant_response=True
        )
        
        dialogue = [system_msg, user_msg, assistant_msg]
        dialogues.append(dialogue)
        
        # Label: 1 = werewolf (colluding), 0 = villager/seer
        is_werewolf = interaction.player_role.lower() == 'werewolf'
        label = Label.HONEST if is_werewolf else Label.DECEPTIVE
        labels.append(label)
    
    return dialogues, labels


def save_dataset_stats(interactions: List[PlayerInteraction], output_path: Path):
    """Save statistics about the extracted dataset."""
    stats = {
        'total_interactions': len(interactions),
        'by_type': {},
        'by_role': {},
        'with_target': 0,
        'same_team': 0,
        'different_team': 0,
        'games': set()
    }
    
    for interaction in interactions:
        # Count by type
        stats['by_type'][interaction.action_type] = stats['by_type'].get(interaction.action_type, 0) + 1
        
        # Count by role
        stats['by_role'][interaction.player_role] = stats['by_role'].get(interaction.player_role, 0) + 1
        
        # Count targets
        if interaction.target:
            stats['with_target'] += 1
            if interaction.same_team is not None:
                if interaction.same_team:
                    stats['same_team'] += 1
                else:
                    stats['different_team'] += 1
        
        stats['games'].add(interaction.game_id)
    
    stats['games'] = sorted(list(stats['games']))
    stats['num_games'] = len(stats['games'])
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Total interactions: {stats['total_interactions']}")
    print(f"  Games: {stats['num_games']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By role: {stats['by_role']}")
    print(f"  With target: {stats['with_target']}")
    print(f"    Same team: {stats['same_team']}")
    print(f"    Different team: {stats['different_team']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract training data from werewolf games")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing game results")
    parser.add_argument("--output-dir", type=str, default="./probe_training_data",
                        help="Output directory for training data")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all interactions
    interactions = extract_all_games(results_dir)
    
    # Save statistics
    save_dataset_stats(interactions, output_dir / "dataset_stats.json")
    
    # Save raw interactions for inspection
    interactions_data = [
        {
            'game_id': i.game_id,
            'turn': i.turn,
            'player': i.player,
            'player_role': i.player_role,
            'action_type': i.action_type,
            'action': i.action,
            'target': i.target,
            'target_role': i.target_role,
            'same_team': i.same_team,
            'has_activations': i.activations is not None
        }
        for i in interactions
    ]
    
    with open(output_dir / "interactions.json", 'w') as f:
        json.dump(interactions_data, f, indent=2)
    
    print(f"\nExtracted {len(interactions)} interactions")
    print(f"Data saved to {output_dir}")
