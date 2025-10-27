"""
Modal backend for serving Apollo deception probes with Llama 70B.
Provides inference API for werewolf game with deception detection.
"""

import modal
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

# Import torch and pickle only when running on Modal (not during local deployment)
if not modal.is_local():
    import torch
    import pickle

# Model configuration
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
N_GPU = 4  # Use 4x H100 for longer context (320GB total)
GPU_CONFIG = f"H100:{N_GPU}"
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
TIMEOUT = 20 * 60  # 20 minutes

# Volume for storing models and detectors
VOLUME = modal.Volume.from_name("werewolf-apollo-probes", create_if_missing=True)
VOLUME_PATH = "/models"
DETECTOR_DIR = Path(VOLUME_PATH) / "detectors"

# Load HF token for accessing Llama models
if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be set in environment or .env file")
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({"HF_TOKEN": hf_token})
else:
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({})

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "vllm==0.6.3",  # Pin to 0.6.3 - last version before V1 engine
        "huggingface_hub>=0.20.0",
        "scikit-learn>=1.3.0",  # For LogisticRegressionDetector
        "jaxtyping>=0.2.0",
    )
)

app = modal.App("werewolf-apollo-probe", image=image)


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={VOLUME_PATH: VOLUME},
    timeout=TIMEOUT,
    secrets=[LOCAL_HF_TOKEN_SECRET],
)
class ApolloProbeService:
    """Modal service for Apollo deception probe inference with Llama 70B."""

    # Use modal.parameter() for class parameterization (Modal 1.0)
    model_name: str = modal.parameter(default=DEFAULT_MODEL)

    # These are initialized in load_model()
    llm: Any = None
    tokenizer: Any = None
    detector: Any = None
    detector_layer: int = 22  # Default layer for roleplaying detector

    @modal.enter()
    def load_model(self):
        """Load vLLM model and detector on container startup."""
        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"Loading vLLM model: {self.model_name}")

        # Initialize vLLM with tensor parallelism for 70B model
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.90,  # 4x H100 = 320GB total, plenty of room
            max_model_len=4096,  # Sufficient for multi-turn werewolf games
            trust_remote_code=True,
            enforce_eager=True,  # Required for hooks
            download_dir=VOLUME_PATH,
            tensor_parallel_size=N_GPU,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN")
        )

        print("Model loaded successfully!")

        # Load default detector if available
        default_detector_path = DETECTOR_DIR / "detector.pt"
        if default_detector_path.exists():
            self._load_detector(str(default_detector_path))
        else:
            print("No default detector found. Use load_detector() to load one.")

    def _load_detector(self, detector_path: str):
        """Load Apollo detector from pickle file."""
        print(f"Loading detector from {detector_path}...")

        with open(detector_path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct detector (simplified version for inference)
        class SimpleDetector:
            def __init__(self, layers, directions, scaler_mean=None, scaler_scale=None,
                         normalize=True, reg_coeff=10.0):
                self.layers = layers if isinstance(layers, list) else [layers]
                # Handle both dict and tensor formats
                if isinstance(directions, dict):
                    self.directions = directions
                else:
                    # Single layer detector - directions is just a tensor
                    self.directions = {self.layers[0]: directions}
                self.scaler_mean = scaler_mean
                self.scaler_scale = scaler_scale
                self.normalize = normalize
                self.reg_coeff = reg_coeff

            def score_activations(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
                """Score activations for a single layer."""
                direction = self.directions[layer].to(activations.device)

                # Ensure direction is 1D for proper matrix multiplication
                if direction.dim() > 1:
                    direction = direction.squeeze()

                # Normalize if needed
                if self.normalize and self.scaler_mean is not None:
                    if isinstance(self.scaler_mean, dict):
                        mean = self.scaler_mean[layer].to(activations.device)
                        scale = self.scaler_scale[layer].to(activations.device)
                    else:
                        # Single layer - scaler is just a tensor
                        mean = self.scaler_mean.to(activations.device)
                        scale = self.scaler_scale.to(activations.device)

                    # Ensure mean/scale are broadcastable
                    if mean.dim() > 1:
                        mean = mean.squeeze()
                    if scale.dim() > 1:
                        scale = scale.squeeze()

                    activations = (activations - mean) / (scale + 1e-8)

                # Project onto direction and apply regularization
                scores = torch.matmul(activations, direction) * self.reg_coeff
                return scores

        self.detector = SimpleDetector(
            layers=data["layers"],
            directions=data["directions"],
            scaler_mean=data.get("scaler_mean"),
            scaler_scale=data.get("scaler_scale"),
            normalize=data.get("normalize", True),
            reg_coeff=data.get("reg_coeff", 10.0)
        )

        # Set the layer to use (use first layer from detector)
        self.detector_layer = data["layers"][0]

        print(f"Detector loaded successfully! Using layer {self.detector_layer}")

    @modal.method()
    def load_detector_from_path(self, detector_path: str) -> Dict[str, Any]:
        """Load a detector from a file path."""
        try:
            self._load_detector(detector_path)
            return {"status": "success", "layer": self.detector_layer}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @modal.method()
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate a response without probe scoring (standard vLLM generation).

        Args:
            messages: Chat messages (e.g., [{"role": "system", "content": "..."}, ...])
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with generated_text
        """
        from vllm import SamplingParams, TokensPrompt

        try:
            # Format the conversation
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9 if temperature > 0 else 1.0,
            )

            # Generate with vLLM (no probe hooks)
            outputs = self.llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
                sampling_params=sampling_params,
                use_tqdm=False
            )

            # Extract generated text
            generated_text = outputs[0].outputs[0].text

            return {
                "generated_text": generated_text,
            }

        except Exception as e:
            import traceback
            return {
                "error": f"Generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    @modal.method()
    def generate_with_probe(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate a response and score it with the probe.

        Args:
            messages: Chat messages (e.g., [{"role": "system", "content": "..."}, ...])
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with generated_text, aggregate_score, and token_scores
        """
        from vllm import SamplingParams

        if self.detector is None:
            return {"error": "No detector loaded. Use load_detector_from_path() first."}

        try:
            # Format the conversation
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9 if temperature > 0 else 1.0,
            )

            # Get the model for hook registration
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            target_layer = model.model.layers[self.detector_layer]

            # Storage for activations
            layer_activations = []
            first_fwd_pass = True

            def activation_hook(module, input, output):
                """Extract activations during generation."""
                nonlocal first_fwd_pass, layer_activations

                if first_fwd_pass:
                    # Skip the first forward pass (prompt processing)
                    first_fwd_pass = False
                    return

                # Extract hidden states
                # vLLM layers return (hidden_states, residual)
                assert len(output) == 2
                hidden_states, residual = output
                resid_post = hidden_states + residual

                # Store the last token's activations
                layer_activations.append(resid_post[-1:].detach().cpu())

            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)

            try:
                # Generate with vLLM
                from vllm import TokensPrompt
                outputs = self.llm.generate(
                    prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

                # Extract generated text
                generated_text = outputs[0].outputs[0].text
                generated_ids = list(outputs[0].outputs[0].token_ids)

            finally:
                # Remove hook
                hook_handle.remove()

            # Score the activations
            if layer_activations:
                # Stack all activations [num_tokens, hidden_dim]
                all_activations = torch.cat(layer_activations, dim=0).to("cuda")

                # Score with detector
                scores = self.detector.score_activations(all_activations, self.detector_layer)
                token_scores = scores.cpu().tolist()

                # Compute aggregate score
                aggregate_score = sum(token_scores) / len(token_scores) if token_scores else 0.5
            else:
                token_scores = []
                aggregate_score = 0.5

            return {
                "generated_text": generated_text,
                "aggregate_score": aggregate_score,
                "token_scores": token_scores,
                "num_tokens": len(generated_ids),
            }

        except Exception as e:
            import traceback
            return {
                "error": f"Generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    @modal.method()
    def generate_with_detailed_probe(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate a response and capture activations at three phases:
        1. Prompt processing (when reading the input)
        2. CoT/reasoning generation
        3. Action/response generation
        
        Returns mean activations for each phase separately.
        """
        from vllm import SamplingParams, TokensPrompt
        
        if self.detector is None:
            return {"error": "No detector loaded."}
        
        try:
            # Format the conversation
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
            
            # Get the model for hook registration
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            target_layer = model.model.layers[self.detector_layer]
            
            # Storage for activations at each phase
            all_activations = []
            num_prompt_tokens = len(prompt_token_ids)
            forward_pass_count = 0
            
            def activation_hook(module, input, output):
                """Extract activations during both prompt and generation."""
                nonlocal forward_pass_count, all_activations
                
                # Extract hidden states
                assert len(output) == 2
                hidden_states, residual = output
                resid_post = hidden_states + residual
                
                # Store all activations
                # First forward pass: processes all prompt tokens at once
                # Subsequent passes: one token at a time during generation
                all_activations.append({
                    'pass_num': forward_pass_count,
                    'activations': resid_post.detach().cpu()
                })
                forward_pass_count += 1
            
            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)
            
            try:
                # Generate response
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9 if temperature > 0 else 1.0,
                )
                
                outputs = self.llm.generate(
                    prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                
                generated_text = outputs[0].outputs[0].text
                
            finally:
                hook_handle.remove()
            
            # Separate prompt and generation activations
            result = {"generated_text": generated_text}
            
            if not all_activations:
                result["prompt_mean_score"] = None
                result["generation_mean_score"] = None
                return result
            
            # First forward pass contains all prompt tokens
            # Subsequent passes are generation (one token each)
            if len(all_activations) > 0:
                # Pass 0: Prompt processing
                prompt_acts = all_activations[0]['activations'].to("cuda")
                prompt_scores = self.detector.score_activations(prompt_acts, self.detector_layer)
                result["prompt_mean_score"] = prompt_scores.mean().item()
                result["prompt_num_tokens"] = len(prompt_acts)
                
                # Passes 1+: Generation (if any)
                if len(all_activations) > 1:
                    gen_acts_list = [a['activations'] for a in all_activations[1:]]
                    gen_acts = torch.cat(gen_acts_list, dim=0).to("cuda")
                    gen_scores = self.detector.score_activations(gen_acts, self.detector_layer)
                    result["generation_mean_score"] = gen_scores.mean().item()
                    result["generation_num_tokens"] = len(gen_acts)
                else:
                    result["generation_mean_score"] = None
                    result["generation_num_tokens"] = 0
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Detailed probe generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    @modal.method()
    def score_prompt_only(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Score a prompt WITHOUT generating any response.
        Just runs the prompt through the model and captures activations.
        
        Perfect for measuring "guilty consciousness" when model learns its role.
        
        Returns:
            Dictionary with prompt_mean_score and prompt_num_tokens
        """
        from vllm import SamplingParams, TokensPrompt
        
        if self.detector is None:
            return {"error": "No detector loaded."}
        
        try:
            # Format the conversation
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False  # Don't add generation prompt - just score the prompt itself
            )
            
            # Get the model for hook registration
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            target_layer = model.model.layers[self.detector_layer]
            
            # Storage for prompt activations
            prompt_activations = []
            
            def activation_hook(module, input, output):
                """Extract activations during prompt processing."""
                nonlocal prompt_activations
                
                # Extract hidden states
                assert len(output) == 2
                hidden_states, residual = output
                resid_post = hidden_states + residual
                
                # Store all activations from this forward pass
                prompt_activations.append(resid_post.detach().cpu())
            
            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)
            
            try:
                # Just do a forward pass through the prompt (no generation)
                # Use generate with max_tokens=1 to get one forward pass
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,  # Only generate 1 token (we'll ignore it)
                )
                
                self.llm.generate(
                    prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                
            finally:
                hook_handle.remove()
            
            # Score the prompt activations
            if prompt_activations:
                # First forward pass contains the prompt
                prompt_acts = prompt_activations[0].to("cuda")
                prompt_scores = self.detector.score_activations(prompt_acts, self.detector_layer)
                
                # Convert to list for JSON serialization
                token_scores = prompt_scores.cpu().tolist()
                
                # Decode tokens for visualization
                tokens = [self.tokenizer.decode([tid]) for tid in prompt_token_ids]
                
                return {
                    "prompt_mean_score": prompt_scores.mean().item(),
                    "prompt_num_tokens": len(prompt_acts),
                    "token_scores": token_scores,
                    "tokens": tokens,
                }
            else:
                return {
                    "prompt_mean_score": None,
                    "prompt_num_tokens": 0,
                    "token_scores": [],
                    "tokens": [],
                }
            
        except Exception as e:
            import traceback
            return {
                "error": f"Prompt scoring failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    @modal.method()
    def score_text(self, text: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        Score existing text for deception (no generation).

        Args:
            text: Text to score
            system_prompt: Optional system prompt

        Returns:
            Dictionary with aggregate_score and token_scores
        """
        from vllm import SamplingParams

        if self.detector is None:
            return {"error": "No detector loaded. Use load_detector_from_path() first."}

        # Build messages - text as assistant message (already generated)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "assistant", "content": text})

        # Format for vLLM
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get model for activation extraction
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        target_layer = model.model.layers[self.detector_layer]

        # Storage for activations
        layer_activations = []

        def hook_fn(module, input, output):
            hidden_states, residual = output
            activations = (hidden_states + residual).clone()
            layer_activations.append(activations)

        # Register hook
        hook = target_layer.register_forward_hook(hook_fn)

        # Use vLLM API with minimal generation to trigger prompt processing
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # Minimal generation - we only want prompt processing
            logprobs=0
        )

        outputs = self.llm.generate(
            formatted_prompt,
            sampling_params,
            use_tqdm=False
        )

        # Remove hook
        hook.remove()

        if not layer_activations:
            return {"error": "Failed to extract activations"}

        # Get activations from prompt processing (first forward pass)
        # vLLM does prefill (prompt) then decode (generation)
        activations = layer_activations[0].squeeze(0)  # Remove batch dim

        # Score the activations
        scores = self.detector.score_activations(
            activations,
            self.detector_layer
        )
        token_scores = scores.cpu().tolist()

        # Compute aggregate
        aggregate_score = sum(token_scores) / len(token_scores) if token_scores else 0.0

        return {
            "aggregate_score": aggregate_score,
            "token_scores": token_scores,
            "num_tokens": len(token_scores),
            "text": text,
        }


@app.function(image=image)
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "werewolf-apollo-probe"}


if __name__ == "__main__":
    # Test the service
    with app.run():
        service = ApolloProbeService()

        # Example: generate with probe
        result = service.generate_with_probe.remote(
            messages=[
                {"role": "system", "content": "You are playing Werewolf. You are a werewolf."},
                {"role": "user", "content": "Who do you think is suspicious?"}
            ],
            max_tokens=100
        )
        print(result)
