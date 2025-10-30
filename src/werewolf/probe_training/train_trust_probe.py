"""
Train trust/collusion probes from werewolf game data.

This script trains probes to detect:
1. Trust/alliance between players (same team vs different teams)
2. Collusion (werewolves vs villagers)

Uses the Apollo deception detection framework with activations extracted from game logs.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional
import argparse

import torch

# Add deception-detection to path
DECEPTION_DETECTION_PATH = Path(__file__).parent.parent.parent.parent / "deception-detection"
sys.path.insert(0, str(DECEPTION_DETECTION_PATH))

from deception_detection.data.base import DialogueDataset, Message
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.activations import Activations
from deception_detection.detectors import LogisticRegressionDetector, get_detector_class
from deception_detection.types import Label

from extract_training_data import (
    extract_all_games,
    create_trust_probe_dataset,
    create_collusion_probe_dataset,
    save_dataset_stats
)


class WerewolfTrustDataset(DialogueDataset):
    """Dataset for trust probe training from werewolf games."""
    
    base_name = "werewolf_trust"
    
    def __init__(
        self,
        results_dir: Path,
        variant: Optional[str] = None,
        model_name: ModelName = ModelName.LLAMA_70B_3_3
    ):
        self.results_dir = results_dir
        self.interactions = extract_all_games(results_dir)
        super().__init__(variant=variant, model_name=model_name)
    
    def _get_dialogues(self):
        """Create dialogues from werewolf interactions."""
        dialogues, labels = create_trust_probe_dataset(self.interactions)
        return dialogues, labels, None


class WerewolfCollusionDataset(DialogueDataset):
    """Dataset for collusion probe training from werewolf games."""
    
    base_name = "werewolf_collusion"
    
    def __init__(
        self,
        results_dir: Path,
        variant: Optional[str] = None,
        model_name: ModelName = ModelName.LLAMA_70B_3_3
    ):
        self.results_dir = results_dir
        self.interactions = extract_all_games(results_dir)
        super().__init__(variant=variant, model_name=model_name)
    
    def _get_dialogues(self):
        """Create dialogues from werewolf interactions."""
        dialogues, labels = create_collusion_probe_dataset(self.interactions)
        return dialogues, labels, None


def train_probe(
    results_dir: Path,
    output_dir: Path,
    probe_type: str = "trust",
    detector_method: str = "lr",
    layers: Optional[List[int]] = None,
    model_name: ModelName = ModelName.LLAMA_70B_3_3,
    batch_size: int = 4,
    max_samples: Optional[int] = None
):
    """
    Train a trust or collusion probe on werewolf game data.
    
    Args:
        results_dir: Directory containing game results
        output_dir: Directory to save trained probe
        probe_type: "trust" or "collusion"
        detector_method: "lr" (logistic regression), "mms", "lat", etc.
        layers: List of model layers to use (default: [22])
        model_name: Which model to use
        batch_size: Batch size for activation extraction
        max_samples: Maximum number of samples to use (for testing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up layers
    if layers is None:
        layers = [22]  # Default to layer 22 like Apollo deception probe
    
    print(f"Training {probe_type} probe using {detector_method} on layers {layers}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("\nLoading dataset...")
    if probe_type == "trust":
        dataset = WerewolfTrustDataset(results_dir, model_name=model_name)
    elif probe_type == "collusion":
        dataset = WerewolfCollusionDataset(results_dir, model_name=model_name)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    print(f"Loaded {len(dataset.dialogues)} dialogues")
    
    # Limit samples if requested
    if max_samples and len(dataset.dialogues) > max_samples:
        print(f"Limiting to {max_samples} samples for testing")
        dataset.dialogues = dataset.dialogues[:max_samples]
        dataset.labels = dataset.labels[:max_samples]
    
    # Save dataset info
    dataset_info = {
        'probe_type': probe_type,
        'num_dialogues': len(dataset.dialogues),
        'label_distribution': {
            'positive': sum(1 for l in dataset.labels if l == Label.HONEST),
            'negative': sum(1 for l in dataset.labels if l == Label.DECEPTIVE)
        },
        'model_name': str(model_name),
        'layers': layers
    }
    
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nLabel distribution:")
    print(f"  Positive (same team / werewolf): {dataset_info['label_distribution']['positive']}")
    print(f"  Negative (diff team / villager): {dataset_info['label_distribution']['negative']}")
    
    # Load model
    print(f"\nLoading model {model_name}...")
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized = dataset.tokenize(tokenizer)
    
    # Split into positive and negative
    positive_indices = [i for i, label in enumerate(dataset.labels) if label == Label.HONEST]
    negative_indices = [i for i, label in enumerate(dataset.labels) if label == Label.DECEPTIVE]
    
    print(f"\nSplitting dataset:")
    print(f"  Positive samples: {len(positive_indices)}")
    print(f"  Negative samples: {len(negative_indices)}")
    
    # Extract activations
    print("\nExtracting activations...")
    print("  Positive samples...")
    positive_tokenized = tokenized.get_indices(positive_indices)
    positive_acts = Activations.from_model(
        model=model,
        tokenized_dataset=positive_tokenized,
        save_logits=False,
        batch_size=batch_size,
        layers=layers,
        verbose=True
    )
    
    print("  Negative samples...")
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
    print(f"\nTraining {detector_method} detector...")
    detector_class = get_detector_class(detector_method)
    detector = detector_class(layers=layers)
    detector.fit(positive_acts, negative_acts)
    
    # Save detector
    detector_path = output_dir / "detector.pt"
    print(f"\nSaving detector to {detector_path}")
    detector.save(detector_path)
    
    # Score on training data for diagnostics
    print("\nScoring on training data...")
    positive_scores = detector.score(positive_acts, all_acts=False)
    negative_scores = detector.score(negative_acts, all_acts=False)
    
    # Compute aggregate statistics
    positive_means = [scores.mean().item() for scores in positive_scores.scores]
    negative_means = [scores.mean().item() for scores in negative_scores.scores]
    
    positive_mean = sum(positive_means) / len(positive_means) if positive_means else 0
    negative_mean = sum(negative_means) / len(negative_means) if negative_means else 0
    
    print(f"\nTraining set scores:")
    print(f"  Positive mean: {positive_mean:.3f}")
    print(f"  Negative mean: {negative_mean:.3f}")
    print(f"  Separation: {positive_mean - negative_mean:.3f}")
    
    # Save training results
    training_results = {
        'probe_type': probe_type,
        'detector_method': detector_method,
        'layers': layers,
        'model_name': str(model_name),
        'num_positive': len(positive_indices),
        'num_negative': len(negative_indices),
        'positive_mean_score': positive_mean,
        'negative_mean_score': negative_mean,
        'separation': positive_mean - negative_mean
    }
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Detector saved to: {detector_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trust/collusion probes on werewolf data")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing game results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trained probe")
    parser.add_argument("--probe-type", type=str, choices=["trust", "collusion"], default="trust",
                        help="Type of probe to train")
    parser.add_argument("--detector-method", type=str, default="lr",
                        choices=["lr", "mms", "lat", "mlp", "cmms"],
                        help="Detection method to use")
    parser.add_argument("--layers", type=int, nargs="+", default=[22],
                        help="Model layers to use for probe")
    parser.add_argument("--model", type=str, default="llama-70b-3.3",
                        help="Model to use (llama-70b-3.3, llama-8b-3.1, etc)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for activation extraction")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples (for testing)")
    
    args = parser.parse_args()
    
    # Map model string to ModelName
    model_map = {
        "llama-70b-3.3": ModelName.LLAMA_70B_3_3,
        "llama-8b-3.1": ModelName.LLAMA_8B_3_1,
    }
    model_name = model_map.get(args.model, ModelName.LLAMA_70B_3_3)
    
    train_probe(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        probe_type=args.probe_type,
        detector_method=args.detector_method,
        layers=args.layers,
        model_name=model_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
