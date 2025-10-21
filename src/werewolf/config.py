"""
Configuration for Werewolf game.
"""

import json
from pathlib import Path
from typing import Optional


class GameConfig:
    """Configuration for Werewolf game."""

    def __init__(self,
                 backend_type: str = "claude",
                 model_url: Optional[str] = None,
                 probe_url: Optional[str] = None,
                 detector_path: Optional[str] = None,
                 provide_probe_scores: bool = False,
                 use_quantized: bool = False,
                 quantization_bits: int = 8,
                 output_dir: str = "/workspace/werewolf_game",
                 max_turns: int = 5):
        """
        Args:
            backend_type: "claude", "local", or "probe"
            model_url: URL for local model server (for "local" backend)
            probe_url: URL for probe server (deprecated, not used)
            detector_path: Path to Apollo detector file (for "probe" backend)
            provide_probe_scores: If True, show probe scores to players as "lie detector"
            use_quantized: If True, load model with quantization (for "probe" backend)
            quantization_bits: 4 or 8 bit quantization (for "probe" backend)
            output_dir: Directory to save game results
            max_turns: Maximum number of game turns
        """
        self.backend_type = backend_type
        self.model_url = model_url or "http://localhost:8000"
        self.probe_url = probe_url or "http://localhost:8001"  # Deprecated
        self.detector_path = detector_path
        self.provide_probe_scores = provide_probe_scores
        self.use_quantized = use_quantized
        self.quantization_bits = quantization_bits
        self.output_dir = output_dir
        self.max_turns = max_turns

    @classmethod
    def from_file(cls, config_path: str) -> 'GameConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "backend_type": self.backend_type,
            "model_url": self.model_url,
            "probe_url": self.probe_url,
            "detector_path": self.detector_path,
            "provide_probe_scores": self.provide_probe_scores,
            "use_quantized": self.use_quantized,
            "quantization_bits": self.quantization_bits,
            "output_dir": self.output_dir,
            "max_turns": self.max_turns
        }

    def save(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
