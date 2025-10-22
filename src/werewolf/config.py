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
                 probe_explanation: Optional[str] = None,
                 request_reasoning: bool = True,
                 score_without_role_context: bool = False,
                 use_quantized: bool = False,
                 quantization_bits: int = 8,
                 modal_app_name: Optional[str] = None,
                 output_dir: str = "/workspace/werewolf_game",
                 max_turns: int = 5,
                 num_players: int = 12,
                 num_werewolves: int = 3,
                 no_seer: bool = False,
                 num_discussion_rounds: int = 1,
                 allow_player_callouts: bool = False):
        """
        Args:
            backend_type: "claude", "local", "probe", or "modal_probe"
            model_url: URL for local model server (for "local" backend)
            probe_url: URL for probe server (deprecated, not used)
            detector_path: Path to Apollo detector file (for "probe" or "modal_probe" backend)
            provide_probe_scores: If True, show probe scores to players as "lie detector"
            probe_explanation: Explanation text for lie detector scores (shown to players)
            request_reasoning: If True, ask models for private reasoning before actions
            score_without_role_context: If True, re-score utterances without role information (ablation study)
            use_quantized: If True, load model with quantization (for "probe" backend)
            quantization_bits: 4 or 8 bit quantization (for "probe" backend)
            modal_app_name: Modal app name (for "modal_probe" backend)
            output_dir: Directory to save game results
            max_turns: Maximum number of game turns
            num_players: Total number of players in the game
            num_werewolves: Number of werewolves in the game
            no_seer: If True, don't include a seer (all non-werewolves are regular villagers)
            num_discussion_rounds: Number of discussion rounds before voting each day
            allow_player_callouts: If True, players can call on others to make additional statements
        """
        self.backend_type = backend_type
        self.model_url = model_url or "http://localhost:8000"
        self.probe_url = probe_url or "http://localhost:8001"  # Deprecated
        self.detector_path = detector_path
        self.provide_probe_scores = provide_probe_scores
        self.probe_explanation = probe_explanation or "Higher scores indicate greater likelihood of deception."
        self.request_reasoning = request_reasoning
        self.score_without_role_context = score_without_role_context
        self.use_quantized = use_quantized
        self.quantization_bits = quantization_bits
        self.modal_app_name = modal_app_name or "werewolf-apollo-probe"
        self.output_dir = output_dir
        self.max_turns = max_turns
        self.num_players = num_players
        self.num_werewolves = num_werewolves
        self.no_seer = no_seer
        self.num_discussion_rounds = num_discussion_rounds
        self.allow_player_callouts = allow_player_callouts

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
            "probe_explanation": self.probe_explanation,
            "request_reasoning": self.request_reasoning,
            "score_without_role_context": self.score_without_role_context,
            "use_quantized": self.use_quantized,
            "quantization_bits": self.quantization_bits,
            "modal_app_name": self.modal_app_name,
            "output_dir": self.output_dir,
            "max_turns": self.max_turns,
            "num_players": self.num_players,
            "num_werewolves": self.num_werewolves,
            "no_seer": self.no_seer,
            "num_discussion_rounds": self.num_discussion_rounds,
            "allow_player_callouts": self.allow_player_callouts
        }

    def save(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
