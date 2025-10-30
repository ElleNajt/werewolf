# Modal Setup for Werewolf Apollo Probes

This guide explains how to use Modal to host the Llama 70B Apollo probes for deception detection in the Werewolf game.

## Why Modal?

- **No local GPU required**: Runs Llama 70B on cloud GPUs (2x H100s)
- **Fast startup**: ~1-2 minutes to load model (vs 10+ minutes on RunPod)
- **Auto-scaling**: Scales down to zero when not in use, scales up on demand
- **Pay-per-use**: Only pay for actual compute time
- **Simple deployment**: One command to deploy

## Prerequisites

1. **Modal account**: Sign up at https://modal.com
2. **Modal CLI**: Install and authenticate:
   ```bash
   pip install modal
   modal setup
   ```

3. **HuggingFace token**: For accessing Llama models
   - Get token from https://huggingface.co/settings/tokens
   - Create `.env` file in project root:
     ```bash
     HF_TOKEN=hf_...
     ```

4. **Modal secrets**: Set HF_TOKEN in Modal dashboard (optional, or use .env)

## Setup Steps

### 1. Deploy the Modal Service

From the `src/werewolf` directory:

```bash
# Deploy the Modal app (creates the service)
modal deploy modal_apollo_backend.py
```

This will:
- Create the Modal app `werewolf-apollo-probe`
- Create a persistent volume `werewolf-apollo-probes` for models
- Deploy the `ApolloProbeService` class

The first deployment takes ~5 minutes to download Llama 70B. Subsequent starts are faster (~1-2 min) thanks to caching.

### 2. Upload the Apollo Detector

Upload the trained detector to the Modal volume:

```bash
# Upload from default location
python deploy_detector_to_modal.py

# Or specify custom path
python deploy_detector_to_modal.py /path/to/detector.pt
```

This uploads the detector to `/models/detectors/detector.pt` on the Modal volume.

### 3. Test the Backend

```bash
# Test the Modal backend integration
python test_modal_backend.py

# Or test direct Modal service call (for debugging)
python test_modal_backend.py --direct
```

If successful, you should see:
- ✓ Connected to Modal service
- ✓ Generated text response
- ✓ Probe deception scores

### 4. Run the Game

Use the Modal backend configuration:

```bash
python game_coordinator.py --config config.modal_probe.json
```

## Configuration

Edit `config.modal_probe.json` to customize:

```json
{
  "backend_type": "modal_probe",
  "modal_app_name": "werewolf-apollo-probe",
  "detector_path": "/models/detectors/detector.pt",
  "provide_probe_scores": true,
  "output_dir": "./werewolf_game_modal",
  "max_turns": 5
}
```

## Cost Estimates

Modal charges for GPU time:
- **H100 GPU**: ~$3-4/hour
- **Typical game**: 5-10 turns × 4 players = 20-40 generations
- **Per generation**: ~2-3 seconds (first generation slower due to startup)
- **Total cost per game**: ~$0.10-0.30

The service auto-scales down after 2 minutes of inactivity (configurable via `scaledown_window` in `modal_apollo_backend.py`).

## Architecture

```
┌─────────────────┐
│  Werewolf Game  │
│  (local)        │
└────────┬────────┘
         │ agent_backend.py
         │ ModalProbeBackend.call()
         ▼
┌─────────────────────────────┐
│  Modal Service              │
│  werewolf-apollo-probe      │
│                             │
│  ┌─────────────────────┐   │
│  │ ApolloProbeService  │   │
│  │                     │   │
│  │ - Llama 70B (vLLM)  │   │
│  │ - Apollo Detector   │   │
│  │ - Activation Hooks  │   │
│  └─────────────────────┘   │
│                             │
│  Volume: /models/           │
│  - detectors/detector.pt    │
│  - cached models            │
└─────────────────────────────┘
```

## Troubleshooting

### "Failed to connect to Modal backend"

- Ensure Modal app is deployed: `modal app list`
- Check app name matches config: `werewolf-apollo-probe`
- Try redeploying: `modal deploy modal_apollo_backend.py`

### "No detector loaded"

- Upload detector: `python deploy_detector_to_modal.py`
- Verify path in config matches: `/models/detectors/detector.pt`
- Check Modal volume: `modal volume list`

### "HF_TOKEN not set"

- Create `.env` file with `HF_TOKEN=hf_...`
- Or set in Modal dashboard: https://modal.com/secrets

### Slow first generation

- First generation takes ~1-2 minutes (model loading + CUDA kernels)
- Subsequent generations are fast (~2-3 seconds)
- This is normal for cold starts

## Advanced Configuration

### Use Multiple Detectors

Upload multiple detectors with different names:

```bash
python deploy_detector_to_modal.py detector_v1.pt
# Then modify upload script to use different remote name
```

Load specific detector:
```python
backend = ModalProbeBackend(
    detector_path="/models/detectors/detector_v1.pt"
)
```

### Adjust GPU Configuration

Edit `modal_apollo_backend.py`:

```python
# Use fewer GPUs (may OOM for 70B)
N_GPU = 1
GPU_CONFIG = "H100:1"

# Use different GPU type
GPU_CONFIG = "A100:2"

# Adjust memory utilization
self.llm = LLM(
    model=self.model_name,
    gpu_memory_utilization=0.95,  # Use more GPU memory
    ...
)
```

### Monitor Usage

View logs in Modal dashboard:
- https://modal.com/apps
- Select `werewolf-apollo-probe`
- View function calls and logs

## Comparison: Modal vs RunPod vs Local

| Aspect | Modal | RunPod | Local |
|--------|-------|--------|-------|
| Setup time | 1 min | 5-10 min | N/A (if no GPU) |
| Cold start | 1-2 min | 0 sec (already running) | N/A |
| Cost | $0.10-0.30/game | $1-2/hour (always on) | Free (if GPU) |
| GPU requirement | None (cloud) | None (cloud) | 2x A100/H100 |
| Auto-scaling | Yes | No | N/A |
| Best for | Development, testing | Long sessions | Not feasible for 70B |

## Next Steps

- **Production deployment**: Use Modal secrets for HF_TOKEN
- **Multiple games**: Modal handles concurrent requests automatically
- **Cost optimization**: Adjust `scaledown_window` based on usage patterns
- **Monitoring**: Set up Modal alerts for errors or high costs
