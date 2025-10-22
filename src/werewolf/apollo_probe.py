"""
Apollo probe integration for deception detection in Werewolf.

This module provides a wrapper around the Apollo deception detection probes
to score statements for deception probability during the game.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Add deception-detection to path
DECEPTION_DETECTION_PATH = Path(__file__).parent.parent.parent / "deception-detection"
sys.path.insert(0, str(DECEPTION_DETECTION_PATH))

from deception_detection.activations import Activations
from deception_detection.detectors import Detector
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.data.base import DialogueDataset, Message
from deception_detection.types import Label


class ApolloProbeWrapper:
    """
    Wrapper for Apollo deception detection probes.

    Handles loading the model, detector, and running inference on text.
    """

    def __init__(
        self,
        detector_path: str,
        model_name: ModelName = ModelName.LLAMA_70B_3_3,
        layers: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the probe wrapper.

        Args:
            detector_path: Path to the trained detector (.pt file)
            model_name: Which Llama model to use
            layers: Which layers to extract activations from (default: [22])
            device: Device to run on
        """
        self.detector_path = Path(detector_path)
        self.model_name = model_name
        self.layers = layers or [22]  # Default to layer 22 from roleplaying probe
        self.device = device

        # Lazy loading - only load when needed
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.detector: Optional[Detector] = None

    def _ensure_loaded(self):
        """Lazy load model and detector."""
        if self.model is None or self.tokenizer is None:
            print(f"Loading model {self.model_name}...")
            self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)
            print("Model loaded.")

        if self.detector is None:
            print(f"Loading detector from {self.detector_path}...")
            # The detector files are pickled, so we can load them directly
            import pickle
            with open(self.detector_path, 'rb') as f:
                data = pickle.load(f)

            # Reconstruct the detector based on saved data
            from deception_detection.detectors import LogisticRegressionDetector
            self.detector = LogisticRegressionDetector(data["layers"])
            self.detector.directions = data["directions"]
            self.detector.scaler_mean = data.get("scaler_mean")
            self.detector.scaler_scale = data.get("scaler_scale")
            self.detector.normalize = data.get("normalize", True)
            self.detector.reg_coeff = data.get("reg_coeff", 10.0)
            print("Detector loaded.")

    def score_text(
        self,
        text: str,
        system_prompt: str = "",
        return_token_scores: bool = False
    ) -> Dict:
        """
        Score a piece of text for deception.

        Args:
            text: The text to score
            system_prompt: Optional system prompt to prepend
            return_token_scores: If True, return per-token scores

        Returns:
            Dictionary with:
                - aggregate_score: Overall deception probability (0-1)
                - token_scores: Per-token scores (if requested)
                - tokens: List of token strings (if requested)
                - text: Original text
        """
        self._ensure_loaded()

        # Create a simple dialogue dataset with one message
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="assistant", content=text))

        dialogue = DialogueDataset(
            dialogues=[messages],
            labels=[Label.UNKNOWN],  # We don't have a label
            dataset_name="werewolf_inference",
            model_name=self.model_name
        )

        # Tokenize
        tokenized = dialogue.tokenize(self.tokenizer)

        # Get activations
        activations = Activations.from_model(
            model=self.model,
            tokenized_dataset=tokenized,
            save_logits=False,
            batch_size=1,
            layers=self.layers,
            verbose=False
        )

        # Score with detector
        scores = self.detector.score(activations, all_acts=False)

        # Extract the scores for the single dialogue
        # scores.scores is a list with one tensor per dialogue
        token_scores_tensor = scores.scores[0]  # Shape: [seq_len]

        # Compute aggregate score (mean of non-nan values)
        valid_scores = token_scores_tensor[~torch.isnan(token_scores_tensor)]
        if len(valid_scores) == 0:
            aggregate = 0.5  # Default if no valid scores
        else:
            aggregate = valid_scores.mean().item()

        result = {
            "aggregate_score": aggregate,
            "text": text,
        }

        if return_token_scores:
            # Get token strings
            tokens_ids = tokenized.tokens[0]  # First (and only) dialogue
            from deception_detection.tokenized_data import get_str_tokens
            str_tokens = get_str_tokens(tokens_ids.unsqueeze(0), self.tokenizer)[0]

            result["tokens"] = str_tokens
            result["token_scores"] = token_scores_tensor.cpu().tolist()

        return result

    def score_batch(
        self,
        texts: List[str],
        system_prompt: str = "",
    ) -> List[Dict]:
        """
        Score multiple texts for deception.

        Args:
            texts: List of texts to score
            system_prompt: Optional system prompt to prepend

        Returns:
            List of score dictionaries (one per text)
        """
        self._ensure_loaded()

        # Create dialogues
        dialogues = []
        for text in texts:
            messages = []
            if system_prompt:
                messages.append(Message(role="system", content=system_prompt))
            messages.append(Message(role="assistant", content=text))
            dialogues.append(messages)

        dialogue_dataset = DialogueDataset(
            dialogues=dialogues,
            labels=[Label.UNKNOWN] * len(texts),
            dataset_name="werewolf_inference",
            model_name=self.model_name
        )

        # Tokenize
        tokenized = dialogue_dataset.tokenize(self.tokenizer)

        # Get activations
        activations = Activations.from_model(
            model=self.model,
            tokenized_dataset=tokenized,
            save_logits=False,
            batch_size=min(8, len(texts)),
            layers=self.layers,
            verbose=False
        )

        # Score with detector
        scores = self.detector.score(activations, all_acts=False)

        # Extract aggregate scores
        results = []
        for i, text in enumerate(texts):
            token_scores_tensor = scores.scores[i]
            valid_scores = token_scores_tensor[~torch.isnan(token_scores_tensor)]

            if len(valid_scores) == 0:
                aggregate = 0.5
            else:
                aggregate = valid_scores.mean().item()

            results.append({
                "aggregate_score": aggregate,
                "text": text,
            })

        return results


def format_probe_reading(
    player_name: str,
    statement: str,
    score_dict: Dict,
    threshold: float = 0.6
) -> str:
    """
    Format probe scores for display to players as a "lie detector" reading.

    Args:
        player_name: Name of the player being scored
        statement: The statement that was scored
        score_dict: Score dictionary from score_text()
        threshold: Threshold above which to flag as high deception

    Returns:
        Formatted string for game context
    """
    aggregate = score_dict["aggregate_score"]

    result = f"\n[LIE DETECTOR] {player_name}: "
    result += f"Score: {aggregate:.2f}"

    return result
