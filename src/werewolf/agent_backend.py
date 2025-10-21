"""
Agent backend abstraction for Werewolf game.
Supports multiple LLM backends with optional deception probe integration.
"""

import subprocess
import json
import requests
import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class AgentBackend(ABC):
    """Base class for agent backends."""

    @abstractmethod
    def call(self, prompt: str, system_prompt: str, max_retries: int = 5) -> Tuple[str, Optional[Dict]]:
        """
        Call the agent backend.

        Returns:
            Tuple of (response_text, metadata)
            metadata can include probe activations, token info, etc.
        """
        pass


class ClaudeBackend(AgentBackend):
    """Backend using claude CLI via subprocess."""

    def call(self, prompt: str, system_prompt: str, max_retries: int = 5) -> Tuple[str, Optional[Dict]]:
        """Call claude -p with retry logic."""
        for attempt in range(max_retries):
            result = subprocess.run(
                ["claude", "-p", prompt, "--system-prompt", system_prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            response = result.stdout.strip()
            # Remove path warnings
            lines = response.split('\n')
            response = '\n'.join([line for line in lines if 'was not found' not in line]).strip()

            # Try to parse JSON
            try:
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()

                # Validate JSON
                json.loads(response)
                return response, None

            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts")
                continue

        raise ValueError("Exhausted retries")


class LocalModelBackend(AgentBackend):
    """Backend using a local Llama model via HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Args:
            base_url: Base URL of the model server (e.g., vLLM, TGI)
        """
        self.base_url = base_url.rstrip('/')

    def call(self, prompt: str, system_prompt: str, max_retries: int = 5) -> Tuple[str, Optional[Dict]]:
        """Call local model API (OpenAI-compatible format)."""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": "default",  # Model name from server
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512,
            "response_format": {"type": "json_object"}  # For JSON mode if supported
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                data = response.json()
                content = data['choices'][0]['message']['content']

                # Validate JSON
                json.loads(content)
                return content, None

            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get valid response: {e}")
                continue

        raise ValueError("Exhausted retries")


class ProbeBackend(AgentBackend):
    """
    Backend with deception probe integration using Apollo probes.

    This backend uses a SINGLE Llama model for both:
    1. Generating player responses
    2. Extracting activations for probe scoring

    This is more memory-efficient than running separate generation + probe servers.
    """

    def __init__(self,
                 detector_path: Optional[str] = None,
                 provide_scores: bool = True,
                 use_quantized: bool = False,
                 quantization_bits: int = 8):
        """
        Args:
            detector_path: Path to Apollo detector (.pt file). If None, uses example probe.
            provide_scores: If True, probe scores are provided to other players as "lie detector"
            use_quantized: If True, load model with quantization (saves memory)
            quantization_bits: 4 or 8 bit quantization (requires bitsandbytes)
        """
        self.provide_scores = provide_scores
        self.use_quantized = use_quantized
        self.quantization_bits = quantization_bits
        self.model = None
        self.tokenizer = None
        self.detector = None
        self.detector_path = detector_path

        # Lazy loading - will load on first call
        print(f"ProbeBackend initialized (quantized={use_quantized})")
        if use_quantized:
            print(f"  Will use {quantization_bits}-bit quantization to reduce memory")
            print(f"  Estimated memory: ~{70 // (2 / (quantization_bits / 8)):.0f}GB + overhead")

    def _ensure_loaded(self):
        """Lazy load model, tokenizer, and detector."""
        if self.model is None or self.tokenizer is None:
            print("Loading Llama 70B model and tokenizer...")

            # Import here to avoid loading if not needed
            from pathlib import Path
            import sys
            DECEPTION_PATH = Path(__file__).parent.parent.parent / "deception-detection"
            if str(DECEPTION_PATH) not in sys.path:
                sys.path.insert(0, str(DECEPTION_PATH))

            from deception_detection.models import ModelName, get_llama3_model_and_tokenizer

            if self.use_quantized:
                # Load with quantization using bitsandbytes
                import torch
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=(self.quantization_bits == 8),
                    load_in_4bit=(self.quantization_bits == 4),
                )

                # Need to modify loading to use quantization
                print(f"Loading with {self.quantization_bits}-bit quantization...")
                # TODO: Update get_llama3_model_and_tokenizer to support quantization
                # For now, fall back to FP16
                print("Warning: Quantization not yet fully implemented, using FP16")
                self.model, self.tokenizer = get_llama3_model_and_tokenizer(
                    ModelName.LLAMA_70B_3_3, omit_model=False
                )
            else:
                self.model, self.tokenizer = get_llama3_model_and_tokenizer(
                    ModelName.LLAMA_70B_3_3, omit_model=False
                )

            print("Model loaded successfully")

        if self.detector is None:
            print("Loading detector...")
            from pathlib import Path
            import pickle

            if self.detector_path is None:
                repo_root = Path(__file__).parent.parent.parent
                self.detector_path = str(repo_root / "deception-detection" / "example_results" / "roleplaying" / "detector.pt")

            with open(self.detector_path, 'rb') as f:
                data = pickle.load(f)

            from deception_detection.detectors import LogisticRegressionDetector
            self.detector = LogisticRegressionDetector(data["layers"])
            self.detector.directions = data["directions"]
            self.detector.scaler_mean = data.get("scaler_mean")
            self.detector.scaler_scale = data.get("scaler_scale")
            self.detector.normalize = data.get("normalize", True)
            self.detector.reg_coeff = data.get("reg_coeff", 10.0)
            print("Detector loaded successfully")

    def call(self, prompt: str, system_prompt: str, max_retries: int = 5) -> Tuple[str, Optional[Dict]]:
        """
        Generate response using Llama and score with probe.

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        self._ensure_loaded()

        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        for attempt in range(max_retries):
            try:
                # Generate response
                # For device_map="auto", don't manually move inputs - model handles it
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
                input_length = inputs['input_ids'].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,  # Try greedy decoding first to diagnose sampling issue
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # Decode only the generated tokens (skip the input prompt)
                response = self.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )

                # Try to extract JSON from response
                response = response.strip()
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()

                # Validate JSON
                parsed = json.loads(response)
                action_text = parsed.get('action', '')

                # Get probe activations for the action text
                if self.provide_scores and action_text:
                    activations = self._get_activations(action_text)
                else:
                    activations = None

                metadata = {
                    "activations": activations,
                }

                return response, metadata

            except (json.JSONDecodeError, KeyError) as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {e}")
                continue

        raise ValueError("Exhausted retries")

    def _get_activations(self, text: str) -> Dict:
        """
        Get probe activations using the already-loaded model and detector.

        Returns:
        {
            "aggregate_score": 0.65,
            "text": "..."
        }
        """
        self._ensure_loaded()

        try:
            import sys
            from pathlib import Path
            DECEPTION_PATH = Path(__file__).parent.parent.parent / "deception-detection"
            if str(DECEPTION_PATH) not in sys.path:
                sys.path.insert(0, str(DECEPTION_PATH))

            from deception_detection.activations import Activations
            from deception_detection.data.base import DialogueDataset, Message
            from deception_detection.types import Label
            from deception_detection.models import ModelName

            # Create a dialogue dataset with one message
            messages = [Message(role="assistant", content=text)]
            dialogue = DialogueDataset(
                dialogues=[messages],
                labels=[Label.UNKNOWN],
                dataset_name="werewolf_inference",
                model_name=ModelName.LLAMA_70B_3_3
            )

            # Tokenize
            tokenized = dialogue.tokenize(self.tokenizer)

            # Get activations using the already-loaded model
            activations = Activations.from_model(
                model=self.model,
                tokenized_dataset=tokenized,
                save_logits=False,
                batch_size=1,
                layers=[22],  # Layer 22 for roleplaying probe
                verbose=False
            )

            # Score with detector
            scores = self.detector.score(activations, all_acts=False)

            # Extract aggregate score
            token_scores_tensor = scores.scores[0]
            valid_scores = token_scores_tensor[~torch.isnan(token_scores_tensor)]

            if len(valid_scores) == 0:
                aggregate = 0.5
            else:
                aggregate = valid_scores.mean().item()

            return {
                "aggregate_score": aggregate,
                "text": text,
            }

        except Exception as e:
            print(f"Error running probe: {e}")
            import traceback
            traceback.print_exc()
            return {
                "aggregate_score": 0.5,
                "text": text,
                "error": str(e)
            }

    def format_activations_for_context(self,
                                      player_name: str,
                                      statement: str,
                                      activations: Dict) -> str:
        """
        Format activation scores for inclusion in game context.

        This presents the scores as a "lie detector" reading to other players.
        """
        if not activations or "error" in activations:
            return f"\n[LIE DETECTOR] {player_name}'s statement: [UNAVAILABLE]"

        aggregate = activations.get("aggregate_score", 0.5)

        result = f"\n[LIE DETECTOR] {player_name}: "
        result += f"Deception probability: {aggregate:.2f}"

        # Flag high/low scores
        if aggregate > 0.6:
            result += " ⚠️ HIGH"
        elif aggregate < 0.4:
            result += " ✓ LOW"

        return result


class ModalProbeBackend(AgentBackend):
    """
    Backend with deception probe integration using Modal for remote inference.

    This backend calls a Modal-hosted Llama 70B service that:
    1. Generates player responses
    2. Extracts activations and scores with Apollo probes

    Much more memory-efficient than running locally, and faster than RunPod.
    """

    def __init__(self,
                 detector_path: Optional[str] = None,
                 provide_scores: bool = True,
                 modal_app_name: str = "werewolf-apollo-probe"):
        """
        Args:
            detector_path: Path to Apollo detector (.pt file) on Modal volume
            provide_scores: If True, probe scores are provided to other players
            modal_app_name: Name of the Modal app to connect to
        """
        self.provide_scores = provide_scores
        self.modal_app_name = modal_app_name
        self.detector_path = detector_path
        self.service = None

        print(f"ModalProbeBackend initialized (app={modal_app_name})")

    def _ensure_connected(self):
        """Lazy connect to Modal service."""
        if self.service is None:
            import modal

            print(f"Connecting to Modal app '{self.modal_app_name}'...")

            # Get the Modal service
            cls = modal.Cls.from_name(self.modal_app_name, "ApolloProbeService")
            self.service = cls()

            # Load detector if path provided
            if self.detector_path:
                result = self.service.load_detector_from_path.remote(self.detector_path)
                if result.get("status") == "error":
                    raise ValueError(f"Failed to load detector: {result.get('message')}")
                print(f"Detector loaded on Modal (layer {result.get('layer')})")

            print("Connected to Modal service!")

    def call(self, prompt: str, system_prompt: str, max_retries: int = 5) -> Tuple[str, Optional[Dict]]:
        """
        Generate response using Modal-hosted Llama with probe scoring.

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        self._ensure_connected()

        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                # Call Modal service
                result = self.service.generate_with_probe.remote(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7
                )

                if "error" in result:
                    raise ValueError(f"Modal service error: {result['error']}")

                generated_text = result["generated_text"]

                # Try to extract JSON from response
                response = generated_text.strip()
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    response = response[json_start:json_end].strip()

                # Validate JSON
                json.loads(response)

                # Build metadata with activations
                metadata = None
                if self.provide_scores:
                    metadata = {
                        "activations": {
                            "aggregate_score": result.get("aggregate_score", 0.5),
                            "text": generated_text,
                        }
                    }

                return response, metadata

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {e}")
                continue

        raise ValueError("Exhausted retries")

    def format_activations_for_context(self,
                                      player_name: str,
                                      statement: str,
                                      activations: Dict) -> str:
        """
        Format activation scores for inclusion in game context.

        This presents the scores as a "lie detector" reading to other players.
        """
        if not activations or "error" in activations:
            return f"\n[LIE DETECTOR] {player_name}'s statement: [UNAVAILABLE]"

        aggregate = activations.get("aggregate_score", 0.5)

        result = f"\n[LIE DETECTOR] {player_name}: "
        result += f"Deception probability: {aggregate:.2f}"

        # Flag high/low scores
        if aggregate > 0.6:
            result += " ⚠️ HIGH"
        elif aggregate < 0.4:
            result += " ✓ LOW"

        return result


def create_backend(backend_type: str = "claude", **kwargs) -> AgentBackend:
    """
    Factory function to create agent backends.

    Args:
        backend_type: One of ["claude", "local", "probe", "modal_probe"]
        **kwargs: Backend-specific configuration

    Returns:
        AgentBackend instance
    """
    if backend_type == "claude":
        return ClaudeBackend()
    elif backend_type == "local":
        return LocalModelBackend(base_url=kwargs.get("model_url", "http://localhost:8000"))
    elif backend_type == "probe":
        return ProbeBackend(
            detector_path=kwargs.get("detector_path", None),
            provide_scores=kwargs.get("provide_scores", True),
            use_quantized=kwargs.get("use_quantized", False),
            quantization_bits=kwargs.get("quantization_bits", 8)
        )
    elif backend_type == "modal_probe":
        return ModalProbeBackend(
            detector_path=kwargs.get("detector_path", None),
            provide_scores=kwargs.get("provide_scores", True),
            modal_app_name=kwargs.get("modal_app_name", "werewolf-apollo-probe")
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
