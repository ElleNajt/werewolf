# Running Trust Probe Training on Modal

## Setup (One Time)

1. **Install Modal**
   ```bash
   pip install modal  # Or: pipx install modal
   ```

2. **Authenticate with Modal**
   ```bash
   modal setup
   ```
   Follow the prompts to connect your Modal account.

## Run Training

```bash
cd src/werewolf/probe_training

modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust
```

## What Happens

1. **Local:** Loads 10 games from `results/experiment_treatment/`
2. **Modal:** Spins up 4x H100 GPUs
3. **Modal:** Loads Llama 70B
4. **Modal:** Extracts werewolf interactions only (~30-45 samples)
5. **Modal:** Trains logistic regression probe on layer 22
6. **Modal:** Returns base64-encoded detector
7. **Local:** Saves detector to `./probes/trust_lr_l22/detector.pt`

**Time:** ~10-15 minutes  
**Cost:** ~$2-3

## Expected Output

```
Loading game data from results/experiment_treatment...
Loaded 10 games

Training trust probe on Modal...
  Detector method: lr
  Layers: [22]
  Batch size: 4

[Modal execution starts]
Training trust probe using lr on layers [22]
Processing 10 games
Extracted 60 werewolf interactions
Created 30 training samples
  Positive: 5    (werewolf → werewolf)
  Negative: 25   (werewolf → villager)
Loading model llama-70b-3.3...
Extracting activations...
  Positive samples: 5
  Negative samples: 25
Training lr detector...

Training results:
  Positive mean: 2.450
  Negative mean: -2.310
  Separation: 4.760

✓ Detector saved locally to: ./probes/trust_lr_l22/detector.pt

{
  "probe_type": "trust",
  "num_positive": 5,
  "num_negative": 25,
  "separation": 4.760,
  "local_path": "./probes/trust_lr_l22/detector.pt"
}
```

## If Modal Is Not Available

You can still prepare the training data and inspect it:

```bash
# Check what data will be used
python3 test_extraction_simple.py --results-dir results/experiment_treatment
```

This shows you have:
- 155 total interactions
- 60 werewolf actions (what we'll use for trust probe)
- 30 werewolf night kills
- ~15-20 werewolf votes

The training script will filter to only werewolf interactions with targets, giving ~30-45 training samples.

## Alternative: Local Training (Requires GPU)

If you have access to a GPU with 70GB+ memory locally:

```bash
python3 train_trust_probe.py \
  --results-dir results/experiment_treatment \
  --output-dir probes/trust_lr_l22 \
  --probe-type trust \
  --layers 22
```

This requires the full PyTorch + transformers environment and a large GPU.

## Troubleshooting

**"modal command not found"**
- Install: `pip install modal` or `pipx install modal`
- May need to use `--break-system-packages` if in managed environment

**"Not authenticated"**
- Run: `modal setup`

**"Not enough training data"**
- You have ~30-45 werewolf interactions with targets
- This is enough for an initial probe
- For better results, run 10-20 more games

**"Imbalanced dataset" (few positive samples)**
- Expected! Werewolves rarely vote for each other
- The probe can still learn from imbalanced data
- Consider: weighting classes or collecting more games
