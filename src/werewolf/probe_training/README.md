# Trust and Collusion Probe Training

This directory contains scripts for training probes to detect trust/alliance and collusion in werewolf games.

## Overview

We train two types of probes:

1. **Trust Probe**: Detects whether a player trusts/allies with another player
   - Label = 1 (positive) if players are on the same team
   - Label = 0 (negative) if players are on different teams
   - Trained on interactions where player A takes an action toward player B

2. **Collusion Probe**: Detects whether a player is part of the colluding team
   - Label = 1 (positive) if player is a werewolf (colluding team)
   - Label = 0 (negative) if player is a villager/seer
   - Trained on all player actions

## Data Extraction

The extraction pipeline processes game logs from `results/` directories and creates training data:

```bash
python extract_training_data.py \
  --results-dir ../../results/experiment_treatment \
  --output-dir ./training_data
```

This extracts:
- Player statements during day discussion
- Vote actions (who voted for whom)
- Night kill actions (werewolf targeting)
- Team labels (same team vs different teams)
- Role labels (werewolf vs villager/seer)

## Probe Training

### Trust Probe

Train a probe to detect trust/alliance between players:

```bash
python train_trust_probe.py \
  --results-dir ../../results/experiment_treatment \
  --output-dir ./probes/trust_probe_lr_l22 \
  --probe-type trust \
  --detector-method lr \
  --layers 22 \
  --batch-size 4
```

### Collusion Probe

Train a probe to detect colluding players (werewolves):

```bash
python train_trust_probe.py \
  --results-dir ../../results/experiment_treatment \
  --output-dir ./probes/collusion_probe_lr_l22 \
  --probe-type collusion \
  --detector-method lr \
  --layers 22 \
  --batch-size 4
```

### Detector Methods

- `lr`: Logistic regression (default, used by Apollo)
- `mms`: Mean-minus-mean separation
- `lat`: Linear Artificial Tomography (RepE method)
- `cmms`: Covariance-adjusted mean separation
- `mlp`: Multi-layer perceptron

### Layer Selection

- Default: Layer 22 (used by Apollo deception probe)
- Can specify multiple layers: `--layers 20 21 22`
- The probe will average scores across layers

## Training Pipeline (GPU Required)

Since probe training requires loading Llama 70B, you must run on a GPU server (e.g., RunPod):

```bash
# 1. Push code to RunPod
runpod push

# 2. Run training on RunPod
runpod run "cd /workspace/src/werewolf/probe_training && \
  python train_trust_probe.py \
    --results-dir /workspace/results/experiment_treatment \
    --output-dir /workspace/probes/trust_probe_$(date +%Y%m%d) \
    --probe-type trust \
    --detector-method lr \
    --layers 22 \
    --batch-size 4"

# 3. Pull trained probe back
runpod pull probes/ ./probes/
```

## Using Trained Probes

Once trained, probes can be loaded and used for inference:

```python
from apollo_probe import ApolloProbeWrapper

# Load trust probe
probe = ApolloProbeWrapper(
    detector_path="./probes/trust_probe_lr_l22/detector.pt",
    layers=[22]
)

# Score a statement
result = probe.score_text(
    text="I think we should vote for Player3",
    system_prompt="You are Player1, a werewolf. How do you feel about Player3?"
)

print(f"Trust score: {result['aggregate_score']:.3f}")
# Higher score = more likely same team (trust)
# Lower score = more likely different teams (distrust)
```

## Output Structure

After training, the output directory contains:

```
probes/trust_probe_lr_l22/
├── detector.pt              # Trained probe (pickle file)
├── dataset_info.json        # Dataset statistics
└── training_results.json    # Training metrics
```

### detector.pt

Pickle file containing:
- `layers`: List of layers used
- `directions`: Probe direction vectors (one per layer)
- `scaler_mean`: Normalization mean (for LR detector)
- `scaler_scale`: Normalization scale (for LR detector)
- `normalize`: Whether to normalize activations
- `reg_coeff`: Regularization coefficient

### dataset_info.json

```json
{
  "probe_type": "trust",
  "num_dialogues": 120,
  "label_distribution": {
    "positive": 60,
    "negative": 60
  },
  "model_name": "llama-70b-3.3",
  "layers": [22]
}
```

### training_results.json

```json
{
  "probe_type": "trust",
  "detector_method": "lr",
  "layers": [22],
  "num_positive": 60,
  "num_negative": 60,
  "positive_mean_score": 2.45,
  "negative_mean_score": -2.31,
  "separation": 4.76
}
```

## Data Requirements

For effective probe training, you need:
- Multiple games (recommended: 10-20+)
- Games with probe activations logged (use `modal_probe` backend)
- Balanced data (roughly equal same-team and different-team interactions)

Games from `results/experiment_treatment/` include activation data from the Apollo deception probe.

## Testing with Small Datasets

For quick testing, limit the number of samples:

```bash
python train_trust_probe.py \
  --results-dir ../../results/experiment_treatment \
  --output-dir ./probes/test_probe \
  --probe-type trust \
  --max-samples 20 \
  --batch-size 2
```

## Expected Results

For a well-trained trust probe:
- Positive samples (same team) should have higher scores
- Negative samples (different teams) should have lower scores
- Good separation: difference of 2-5+ between means
- Balanced dataset helps: aim for similar numbers of positive/negative

## Troubleshooting

**Issue: "No interactions with targets found"**
- Make sure games were run with `provide_probe_scores: true`
- Check that `player_activations` is present in `game_stats.json`

**Issue: "Imbalanced dataset"**
- This is expected in werewolf (more villagers than werewolves)
- For trust probe: ensure both same-team and cross-team interactions exist
- For collusion probe: imbalance is inherent to the game

**Issue: "Poor separation"**
- Try different detector methods (`--detector-method`)
- Try different layers (`--layers 20 21 22`)
- Collect more training data (run more games)

## Next Steps

After training probes:

1. **Evaluate on held-out games**: Train on some games, test on others
2. **Test generalization**: Do probes trained on werewolf work in other games?
3. **Integrate into gameplay**: Use trust probes during games to detect alliances
4. **Analyze probe directions**: What features do they capture?
