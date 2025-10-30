# Quick Start: Werewolf with Modal

Get started with Werewolf + Apollo probes in under 10 minutes, no GPU required.

## 1. Prerequisites (5 min)

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup

# Install werewolf dependencies
pip install -r ../../requirements.txt

# Get HuggingFace token from https://huggingface.co/settings/tokens
# Create .env file in project root
echo "HF_TOKEN=hf_your_token_here" > .env
```

## 2. Deploy to Modal (2 min first time, 30 sec after)

```bash
cd src/werewolf

# Deploy the Modal service
modal deploy modal_apollo_backend.py
```

This deploys Llama 70B to Modal. First time takes ~2 minutes to download the model.

## 3. Upload Detector (30 sec)

```bash
# Upload the Apollo detector to Modal
python deploy_detector_to_modal.py

# Or specify custom path
python deploy_detector_to_modal.py /path/to/detector.pt
```

## 4. Test (1 min)

```bash
# Test the backend
python test_modal_backend.py
```

Expected output:
```
=== Testing Modal Apollo Probe Backend ===

1. Creating ModalProbeBackend...
ModalProbeBackend initialized (app=werewolf-apollo-probe)

2. Testing generation with probe scoring...
Connecting to Modal app 'werewolf-apollo-probe'...
Connected to Modal service!

✓ Generation successful!

Response JSON:
{"action": "I think we should focus on behavior patterns..."}

Probe scores:
  Aggregate deception score: 0.423

Formatted for game:
[LIE DETECTOR] TestPlayer: Deception probability: 0.42 ✓ LOW

=== All tests passed! ===
```

## 5. Run a Game! (1-2 min per game)

```bash
# Run with Modal backend
python game_coordinator.py config.modal_probe.json
```

Results will be saved to `./werewolf_game_modal/`.

## Troubleshooting

### "Failed to connect to Modal backend"
- Check Modal app exists: `modal app list`
- Should see: `werewolf-apollo-probe`
- If not, redeploy: `modal deploy modal_apollo_backend.py`

### "HF_TOKEN not set"
- Create `.env` file with `HF_TOKEN=hf_...`
- Get token from: https://huggingface.co/settings/tokens

### "No detector loaded"
- Run: `python deploy_detector_to_modal.py`
- Verify upload succeeded (should see "✓ Detector uploaded successfully")

### Slow first generation
- First generation takes ~1-2 min (model loading)
- This is normal for cold starts
- Subsequent generations are fast (~2-3 sec)

## Cost

Typical costs for Modal usage:
- **First game of session**: ~$0.20 (includes cold start)
- **Subsequent games**: ~$0.10 each (model already loaded)
- **Idle time**: $0 (auto-scales to zero after 5 min)

For a 1-hour session with ~5 games: **~$0.60 total**

## What's Next?

- **Adjust game settings**: Edit `config.modal_probe.json`
- **Run multiple games**: Modal handles concurrency automatically
- **Monitor usage**: https://modal.com/apps → `werewolf-apollo-probe`
- **View logs**: Click on function calls in Modal dashboard

## Advanced Usage

### Use Different Detector

```bash
# Upload with custom name
python deploy_detector_to_modal.py my_detector.pt

# Update config.modal_probe.json
{
  "detector_path": "/models/detectors/my_detector.pt"
}
```

### Adjust GPU Configuration

Edit `modal_apollo_backend.py`:
```python
N_GPU = 1  # Use fewer GPUs (may OOM)
GPU_CONFIG = "A100:2"  # Use A100s instead of H100s
```

### Monitor in Real-time

```bash
# Watch logs as game runs
modal app logs werewolf-apollo-probe --follow
```

## Comparison to Other Backends

| Backend | Setup Time | Cost/Game | GPU Needed | Best For |
|---------|------------|-----------|------------|----------|
| Modal | 5 min | $0.10-0.20 | No | **Most users** |
| RunPod | 10+ min | $1-2/hour | No | Long sessions |
| Local | N/A | Free | Yes (2xH100) | You have GPUs |
| Claude | 1 min | Free* | No | Testing game logic |

*Claude backend doesn't include probes

**Recommendation**: Use Modal for development and small-scale experiments.
