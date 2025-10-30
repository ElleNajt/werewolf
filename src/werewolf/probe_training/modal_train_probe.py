"""
Modal deployment for training trust/collusion probes.

This allows training probes in the cloud without managing GPU infrastructure.
"""

import modal
import json
from pathlib import Path
from typing import List, Optional, Dict

# Modal configuration
GPU_CONFIG = "H100:4"  # 4x H100 for Llama 70B
TIMEOUT = 60 * 60  # 1 hour
VOLUME = modal.Volume.from_name("werewolf-probe-training", create_if_missing=True)
VOLUME_PATH = "/data"

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Install git first
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
        "scikit-learn>=1.3.0",
        "jaxtyping>=0.2.0",
    )
    .run_commands(
        "git clone https://github.com/anthropics/deception-detection /root/deception-detection || true"
    )
    .env({"PYTHONPATH": "/root/deception-detection"})  # Don't use $PYTHONPATH expansion
)

app = modal.App("werewolf-probe-training", image=image)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes={VOLUME_PATH: VOLUME},
)
def train_probe_modal(
    game_data: List[Dict],
    probe_type: str = "trust",
    detector_method: str = "lr",
    layers: Optional[List[int]] = None,
    model_name: str = "llama-70b-3.3",
    batch_size: int = 4,
) -> Dict:
    """
    Train a trust or collusion probe on werewolf game data.
    
    Args:
        game_data: List of game stats dictionaries
        probe_type: "trust" or "collusion"
        detector_method: "lr", "mms", "lat", etc.
        layers: List of layers to use (default: [22])
        model_name: Model to use
        batch_size: Batch size for activation extraction
        
    Returns:
        Dictionary with training results and detector data
    """
    import sys
    sys.path.insert(0, "/root/deception-detection")
    
    import torch
    from deception_detection.data.base import DialogueDataset, Message
    from deception_detection.models import ModelName, get_model_and_tokenizer
    from deception_detection.activations import Activations
    from deception_detection.detectors import get_detector_class
    from deception_detection.types import Label
    
    # Set layers
    if layers is None:
        layers = [22]
    
    print(f"Training {probe_type} probe using {detector_method} on layers {layers}")
    print(f"Processing {len(game_data)} games")
    
    # Extract interactions from game data
    from extract_interactions import extract_interactions_from_stats, are_same_team
    
    all_interactions = []
    for game_stats in game_data:
        interactions = extract_interactions_from_stats(game_stats)
        all_interactions.extend(interactions)
    
    print(f"Extracted {len(all_interactions)} interactions")
    
    # Create dialogues based on probe type
    dialogues = []
    labels = []
    
    if probe_type == "trust":
        # Trust probe: ONLY werewolf interactions (only they have ground truth)
        # Werewolves know who their allies are; villagers don't
        for interaction in all_interactions:
            if interaction['target'] is None or interaction['same_team'] is None:
                continue
            
            # CRITICAL: Only use werewolf interactions
            if interaction['player_role'].lower() != 'werewolf':
                continue
            
            system_msg = Message(
                role="system",
                content=f"You are {interaction['player']}, a {interaction['player_role']} in a Werewolf game.",
                is_assistant_response=False
            )
            user_msg = Message(
                role="user",
                content=f"How do you feel about {interaction['target']}?",
                is_assistant_response=False
            )
            assistant_msg = Message(
                role="assistant",
                content=interaction['action'],
                is_assistant_response=True
            )
            
            dialogues.append([system_msg, user_msg, assistant_msg])
            # Label: 1 = ally (trust), 0 = enemy (distrust)
            label = Label.HONEST if interaction['same_team'] else Label.DECEPTIVE
            labels.append(label)
    
    elif probe_type == "collusion":
        # Collusion probe: werewolf vs villager
        for interaction in all_interactions:
            system_msg = Message(
                role="system",
                content=f"You are {interaction['player']} in a Werewolf game.",
                is_assistant_response=False
            )
            user_msg = Message(
                role="user",
                content="What do you say?",
                is_assistant_response=False
            )
            assistant_msg = Message(
                role="assistant",
                content=interaction['action'],
                is_assistant_response=True
            )
            
            dialogues.append([system_msg, user_msg, assistant_msg])
            is_werewolf = interaction['player_role'].lower() == 'werewolf'
            label = Label.HONEST if is_werewolf else Label.DECEPTIVE
            labels.append(label)
    
    print(f"Created {len(dialogues)} training samples")
    print(f"  Positive: {sum(1 for l in labels if l == Label.HONEST)}")
    print(f"  Negative: {sum(1 for l in labels if l == Label.DECEPTIVE)}")
    
    # Create dataset
    class WerewolfDataset(DialogueDataset):
        base_name = f"werewolf_{probe_type}"
        
        def __init__(self, dialogues, labels):
            self._dialogues = dialogues
            self._labels = labels
            self.model_name = ModelName.LLAMA_70B_3_3
        
        def _get_dialogues(self):
            return self._dialogues, self._labels, None
    
    dataset = WerewolfDataset(dialogues, labels)
    
    # Load model
    print(f"Loading model {model_name}...")
    model_enum = ModelName.LLAMA_70B_3_3 if "70b" in model_name.lower() else ModelName.LLAMA_8B_3_1
    model, tokenizer = get_model_and_tokenizer(model_enum)
    
    # Tokenize
    print("Tokenizing...")
    tokenized = dataset.tokenize(tokenizer)
    
    # Split positive/negative
    positive_indices = [i for i, label in enumerate(labels) if label == Label.HONEST]
    negative_indices = [i for i, label in enumerate(labels) if label == Label.DECEPTIVE]
    
    print(f"Extracting activations...")
    print(f"  Positive samples: {len(positive_indices)}")
    positive_tokenized = tokenized.get_indices(positive_indices)
    positive_acts = Activations.from_model(
        model=model,
        tokenized_dataset=positive_tokenized,
        save_logits=False,
        batch_size=batch_size,
        layers=layers,
        verbose=True
    )
    
    print(f"  Negative samples: {len(negative_indices)}")
    negative_tokenized = tokenized.get_indices(negative_indices)
    negative_acts = Activations.from_model(
        model=model,
        tokenized_dataset=negative_tokenized,
        save_logits=False,
        batch_size=batch_size,
        layers=layers,
        verbose=True
    )
    
    # Train detector
    print(f"Training {detector_method} detector...")
    detector_class = get_detector_class(detector_method)
    detector = detector_class(layers=layers)
    detector.fit(positive_acts, negative_acts)
    
    # Score training data
    print("Computing training scores...")
    positive_scores = detector.score(positive_acts, all_acts=False)
    negative_scores = detector.score(negative_acts, all_acts=False)
    
    positive_means = [scores.mean().item() for scores in positive_scores.scores]
    negative_means = [scores.mean().item() for scores in negative_scores.scores]
    
    positive_mean = sum(positive_means) / len(positive_means) if positive_means else 0
    negative_mean = sum(negative_means) / len(negative_means) if negative_means else 0
    
    print(f"\nTraining results:")
    print(f"  Positive mean: {positive_mean:.3f}")
    print(f"  Negative mean: {negative_mean:.3f}")
    print(f"  Separation: {positive_mean - negative_mean:.3f}")
    
    # Save detector to volume
    import pickle
    detector_data = {
        "layers": layers,
        "directions": detector.directions.cpu() if detector.directions is not None else None,
    }
    
    if hasattr(detector, 'scaler_mean') and detector.scaler_mean is not None:
        detector_data.update({
            "scaler_mean": detector.scaler_mean.cpu(),
            "scaler_scale": detector.scaler_scale.cpu(),
            "normalize": detector.normalize,
            "reg_coeff": detector.reg_coeff,
        })
    
    # Serialize detector for return
    import base64
    detector_bytes = pickle.dumps(detector_data)
    detector_b64 = base64.b64encode(detector_bytes).decode('utf-8')
    
    # Also save to volume as backup
    output_dir = Path(VOLUME_PATH) / "trained_probes" / f"{probe_type}_{detector_method}_l{'_'.join(map(str, layers))}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector_path = output_dir / "detector.pt"
    with open(detector_path, 'wb') as f:
        pickle.dump(detector_data, f)
    
    VOLUME.commit()
    
    print(f"\nDetector saved to volume: {output_dir}")
    print(f"Detector size: {len(detector_bytes)} bytes")
    
    # Return results WITH serialized detector
    return {
        "probe_type": probe_type,
        "detector_method": detector_method,
        "layers": layers,
        "num_games": len(game_data),
        "num_interactions": len(all_interactions),
        "num_positive": len(positive_indices),
        "num_negative": len(negative_indices),
        "positive_mean_score": positive_mean,
        "negative_mean_score": negative_mean,
        "separation": positive_mean - negative_mean,
        "detector_path": str(detector_path),
        "detector_data": detector_b64,  # Base64-encoded detector for direct download
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
)
def list_trained_probes() -> List[str]:
    """List all trained probes in the volume."""
    probes_dir = Path(VOLUME_PATH) / "trained_probes"
    if not probes_dir.exists():
        return []
    
    return [str(p.relative_to(probes_dir)) for p in probes_dir.iterdir() if p.is_dir()]


@app.function(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
)
def download_probe(probe_name: str) -> bytes:
    """Download a trained probe."""
    probe_path = Path(VOLUME_PATH) / "trained_probes" / probe_name / "detector.pt"
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_name}")
    
    with open(probe_path, 'rb') as f:
        return f.read()


# Standalone helper for extracting interactions (needed by Modal function)
@app.function(image=image)
def extract_interactions():
    """Helper module for extracting interactions from game stats."""
    code = """
def are_same_team(role1: str, role2: str) -> bool:
    werewolf_team = {'werewolf'}
    village_team = {'villager', 'seer'}
    
    role1_lower = role1.lower()
    role2_lower = role2.lower()
    
    if role1_lower in werewolf_team and role2_lower in werewolf_team:
        return True
    if role1_lower in village_team and role2_lower in village_team:
        return True
    return False


def extract_interactions_from_stats(stats):
    interactions = []
    
    roles = {}
    for player_data in stats['players']:
        roles[player_data['name']] = player_data['role']
    
    game_id = stats['game_id']
    
    for event in stats['history']:
        turn = event['turn']
        event_type = event['type']
        data = event['data']
        
        if event_type == 'day_statement':
            player = data['player']
            statement = data['statement']
            player_role = roles[player]
            
            interactions.append({
                'game_id': game_id,
                'turn': turn,
                'player': player,
                'player_role': player_role,
                'action_type': 'statement',
                'action': statement,
                'target': None,
                'target_role': None,
                'same_team': None
            })
        
        elif event_type == 'day_elimination':
            # Find who voted for whom from activations
            player_acts = stats.get('player_activations', {})
            for player, acts in player_acts.items():
                if player not in roles:
                    continue
                player_role = roles[player]
                
                for act in acts:
                    if act.get('action') in roles.keys():
                        voted_for = act['action']
                        voted_for_role = roles[voted_for]
                        same_team = are_same_team(player_role, voted_for_role)
                        
                        interactions.append({
                            'game_id': game_id,
                            'turn': turn,
                            'player': player,
                            'player_role': player_role,
                            'action_type': 'vote',
                            'action': voted_for,
                            'target': voted_for,
                            'target_role': voted_for_role,
                            'same_team': same_team
                        })
                        break
        
        elif event_type == 'night_kill':
            werewolf_votes = data.get('werewolf_votes', {})
            for werewolf, target in werewolf_votes.items():
                werewolf_role = roles[werewolf]
                target_role = roles.get(target)
                
                if target_role:
                    same_team = are_same_team(werewolf_role, target_role)
                    
                    interactions.append({
                        'game_id': game_id,
                        'turn': turn,
                        'player': werewolf,
                        'player_role': werewolf_role,
                        'action_type': 'night_kill',
                        'action': target,
                        'target': target,
                        'target_role': target_role,
                        'same_team': same_team
                    })
    
    return interactions
"""
    # Write to a temp file in the image
    with open("/tmp/extract_interactions.py", 'w') as f:
        f.write(code)
    
    return code


if __name__ == "__main__":
    # For local testing - load game data and train
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: modal run modal_train_probe.py <results_dir> [--probe-type trust|collusion]")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    probe_type = "trust"
    
    if "--probe-type" in sys.argv:
        idx = sys.argv.index("--probe-type")
        probe_type = sys.argv[idx + 1]
    
    # Load game data
    game_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('game')])
    game_data = []
    
    for game_dir in game_dirs:
        stats_file = game_dir / "game_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                game_data.append(json.load(f))
    
    print(f"Loaded {len(game_data)} games")
    
    # Train probe on Modal
    with app.run():
        result = train_probe_modal.remote(
            game_data=game_data,
            probe_type=probe_type,
            detector_method="lr",
            layers=[22],
            batch_size=4
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(json.dumps(result, indent=2))
