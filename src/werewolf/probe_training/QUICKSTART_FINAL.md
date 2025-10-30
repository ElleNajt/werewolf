# Trust Probe Training - Quick Start

## Current Location
You're in a container at `/workspace`

## Setup (One Time)

1. **Authenticate with Modal** (interactive):
   ```bash
   cd /workspace
   probe_venv/bin/modal token new
   ```
   This opens a browser to sign in to Modal.

## Run Training

From anywhere in the container:

```bash
cd /workspace/src/werewolf/probe_training
./run_training.sh
```

Or manually:

```bash
cd /workspace/src/werewolf/probe_training
../../../probe_venv/bin/modal run train_on_modal.py train \
  --results-dir ../../../results/experiment_treatment \
  --probe-type trust
```

## What This Does

**Training a werewolf trust probe:**
- Uses only werewolf interactions (they know who allies are)
- Filters to ~30-45 samples from 10 games
- Trains on Modal (4x H100 GPUs, ~10-15 min, ~$2-3)
- Saves probe to: `./probes/trust_lr_l22/detector.pt`

**The probe learns:** Neural patterns that distinguish when a werewolf views another player as ally (high score) vs enemy (low score).

## Your Data

From `results/experiment_treatment/`:
- ✅ 10 games with activation data
- ✅ 60 werewolf actions total
- ✅ ~30 night kills (werewolf → villager)
- ✅ ~15-20 votes (mostly werewolf → villager, few werewolf → werewolf)

**Training samples:**
- Positive (ally): ~5-10 werewolf → werewolf interactions
- Negative (enemy): ~25-35 werewolf → villager interactions

## Expected Output

```
Loading game data from ../../../results/experiment_treatment...
Loaded 10 games

Training trust probe on Modal...
[Modal spins up 4x H100]
Extracted 60 werewolf interactions
Created 30 training samples
  Positive: 5    (werewolf → werewolf = trust)
  Negative: 25   (werewolf → villager = distrust)

Training lr detector...
  Positive mean: 2.45
  Negative mean: -2.31
  Separation: 4.76

✓ Detector saved locally to: ./probes/trust_lr_l22/detector.pt
```

## File Structure

```
/workspace/
├── probe_venv/                        # Modal installed here
├── results/
│   └── experiment_treatment/          # Your 10 games
└── src/werewolf/probe_training/
    ├── run_training.sh                # Run this!
    ├── probes/
    │   └── trust_lr_l22/
    │       └── detector.pt            # ← Trained probe output
    └── ...
```

## Test Your Data First

```bash
cd /workspace/src/werewolf/probe_training
python3 test_extraction_simple.py --results-dir ../../../results/experiment_treatment
```

Shows you have:
- 155 total interactions
- 60 werewolf actions (what we use)
- 190 activations available

## Troubleshooting

**"Token missing"**
```bash
cd /workspace
probe_venv/bin/modal token new
```

**"Not enough data"**
- You have enough (~30 werewolf samples)
- For more, run additional games

**"Imbalanced dataset"**
- Expected! Werewolves rarely vote for each other
- Probe can still learn from imbalanced data

## Using the Trained Probe

```python
import sys
sys.path.insert(0, '/workspace/src/werewolf')
from apollo_probe import ApolloProbeWrapper

probe = ApolloProbeWrapper(
    detector_path="./probes/trust_lr_l22/detector.pt",
    layers=[22]
)

# Werewolf thinking about ally
result = probe.score_text(
    text="I think Player5 is trustworthy",
    system_prompt="You are Player3, a werewolf. Player5 is also a werewolf."
)
print(f"Trust (ally): {result['aggregate_score']:.3f}")  # Expect: high

# Werewolf thinking about enemy  
result = probe.score_text(
    text="I think Player1 is suspicious",
    system_prompt="You are Player3, a werewolf. Player1 is a villager."
)
print(f"Trust (enemy): {result['aggregate_score']:.3f}")  # Expect: low
```

## All Set!

Everything is ready. Just run:
```bash
cd /workspace/src/werewolf/probe_training
./run_training.sh
```

(After authenticating with `probe_venv/bin/modal token new`)
