#!/usr/bin/env python3
"""
Simple script to train probes using Modal.

This is a cleaner interface than the full modal_train_probe.py
"""

import modal
import json
import sys
from pathlib import Path
from typing import List, Dict

# Import the Modal app
app_lookup = modal.App.lookup("werewolf-probe-training", create_if_missing=False)


def load_game_data(results_dir: Path) -> List[Dict]:
    """Load all game stats from a results directory."""
    game_dirs = sorted([d for d in results_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('game')])
    
    game_data = []
    for game_dir in game_dirs:
        stats_file = game_dir / "game_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                game_data.append(json.load(f))
    
    return game_data


def train_probe(
    results_dir: str,
    probe_type: str = "trust",
    detector_method: str = "lr",
    layers: List[int] = None,
    batch_size: int = 4,
    output_path: str = None,
):
    """
    Train a probe using Modal.
    
    Args:
        results_dir: Path to directory containing game results
        probe_type: "trust" or "collusion"
        detector_method: "lr", "mms", "lat", etc.
        layers: List of layers to use (default: [22])
        batch_size: Batch size for activation extraction
        output_path: Where to save the trained detector locally
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return
    
    print(f"Loading game data from {results_dir}...")
    game_data = load_game_data(results_path)
    
    if not game_data:
        print("Error: No game data found!")
        return
    
    print(f"Loaded {len(game_data)} games")
    print(f"\nTraining {probe_type} probe on Modal...")
    print(f"  Detector method: {detector_method}")
    print(f"  Layers: {layers or [22]}")
    print(f"  Batch size: {batch_size}")
    
    # Get the training function from Modal
    train_fn = app_lookup.registered_functions["train_probe_modal"]
    
    # Run training
    result = train_fn.remote(
        game_data=game_data,
        probe_type=probe_type,
        detector_method=detector_method,
        layers=layers,
        batch_size=batch_size
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Extract and save detector locally
    if "detector_data" in result:
        import base64
        import pickle
        
        detector_b64 = result.pop("detector_data")
        detector_bytes = base64.b64decode(detector_b64)
        
        # Determine output path
        if output_path is None:
            layers_str = "_".join(map(str, layers or [22]))
            output_path = f"./probes/{probe_type}_{detector_method}_l{layers_str}/detector.pt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(detector_bytes)
        
        print(f"\n✓ Detector saved locally to: {output_file}")
        result["local_path"] = str(output_file)
    
    print(json.dumps(result, indent=2))
    
    return result


def list_probes():
    """List all trained probes stored in Modal volume."""
    list_fn = app_lookup.registered_functions["list_trained_probes"]
    probes = list_fn.remote()
    
    print("Trained probes in Modal volume:")
    for probe in probes:
        print(f"  - {probe}")
    
    return probes


def download_probe(probe_name: str, output_path: str):
    """Download a trained probe from Modal volume."""
    download_fn = app_lookup.registered_functions["download_probe"]
    
    print(f"Downloading probe: {probe_name}")
    probe_data = download_fn.remote(probe_name)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        f.write(probe_data)
    
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train probes using Modal")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a probe")
    train_parser.add_argument("--results-dir", type=str, required=True,
                             help="Directory containing game results")
    train_parser.add_argument("--probe-type", type=str, default="trust",
                             choices=["trust", "collusion"],
                             help="Type of probe to train")
    train_parser.add_argument("--detector-method", type=str, default="lr",
                             help="Detection method (lr, mms, lat, etc.)")
    train_parser.add_argument("--layers", type=int, nargs="+", default=[22],
                             help="Model layers to use")
    train_parser.add_argument("--batch-size", type=int, default=4,
                             help="Batch size for activation extraction")
    train_parser.add_argument("--output", type=str, default=None,
                             help="Output path for detector (default: ./probes/{type}_{method}_l{layers}/detector.pt)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List trained probes")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a trained probe")
    download_parser.add_argument("probe_name", type=str,
                                help="Name of probe to download")
    download_parser.add_argument("--output", type=str, default="./detector.pt",
                                help="Output path for detector file")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_probe(
            results_dir=args.results_dir,
            probe_type=args.probe_type,
            detector_method=args.detector_method,
            layers=args.layers,
            batch_size=args.batch_size,
            output_path=args.output
        )
    elif args.command == "list":
        list_probes()
    elif args.command == "download":
        download_probe(args.probe_name, args.output)
    else:
        parser.print_help()
