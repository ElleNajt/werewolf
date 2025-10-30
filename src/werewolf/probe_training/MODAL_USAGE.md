# Training Probes on Modal

## Quick Answer: Where Does Modal Store the Probe?

**Two places:**

1. **Returned directly to you** (recommended): The detector is base64-encoded in the result and automatically saved locally
2. **Modal Volume backup**: Also saved to Modal's persistent storage at `/data/trained_probes/`

When you run training, the probe is **automatically saved locally** to:
```
./probes/{probe_type}_{detector_method}_l{layers}/detector.pt
```

Example: `./probes/trust_lr_l22/detector.pt`

## Usage

### 1. Train a Probe

```bash
cd src/werewolf/probe_training

# Train trust probe (detects same-team vs different-team)
modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust \
  --layers 22

# Train collusion probe (detects werewolf vs villager)
modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type collusion \
  --layers 22
```

The probe will be:
- ✅ Trained on Modal's GPUs (4x H100)
- ✅ Automatically downloaded and saved locally to `./probes/`
- ✅ Also backed up to Modal volume

### 2. Custom Output Path

```bash
modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust \
  --output my_probes/trust_probe.pt
```

### 3. List Probes in Modal Volume

```bash
modal run train_on_modal.py list
```

This shows probes backed up in the Modal volume (not needed if you saved locally).

### 4. Download from Modal Volume

If you need to re-download a probe from the backup:

```bash
modal run train_on_modal.py download trust_lr_l22 \
  --output ./probes/trust_probe.pt
```

## How It Works

**Training Flow:**

1. Your local script loads game data from `results/` directory
2. Game data is sent to Modal (serialized JSON)
3. Modal:
   - Loads Llama 70B on 4x H100 GPUs
   - Extracts activations from game interactions
   - Trains logistic regression probe
   - Serializes detector to base64
   - Saves backup to Modal volume
   - **Returns detector data in the result**
4. Local script:
   - Receives base64-encoded detector
   - Decodes and saves to `./probes/detector.pt`

**Result:** You get the trained probe locally without manually downloading!

## Storage Breakdown

### Local Storage (Automatic)
```
src/werewolf/probe_training/probes/
├── trust_lr_l22/
│   └── detector.pt          # ← Automatically saved here
└── collusion_lr_l22/
    └── detector.pt
```

### Modal Volume (Backup)
```
Modal Volume: "werewolf-probe-training"
/data/trained_probes/
├── trust_lr_l22/
│   └── detector.pt
└── collusion_lr_l22/
    └── detector.pt
```

The Modal volume is useful if:
- You want to share probes between Modal runs
- You need to re-download a probe later
- You're running entirely in Modal (no local download)

## Complete Example

```bash
# 1. Check your data
python3 test_extraction_simple.py --results-dir results/experiment_treatment

# 2. Train trust probe on Modal
modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust \
  --detector-method lr \
  --layers 22 \
  --batch-size 4

# Output shows:
#   Training on Modal...
#   Extracting activations...
#   Training lr detector...
#   ✓ Detector saved locally to: ./probes/trust_lr_l22/detector.pt

# 3. Use the probe
ls -lh ./probes/trust_lr_l22/detector.pt
```

## Advanced Options

### Multi-Layer Probes

```bash
modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust \
  --layers 20 21 22
```

Saves to: `./probes/trust_lr_l20_21_22/detector.pt`

### Different Detection Methods

```bash
# Mean-minus-mean separation
modal run train_on_modal.py train \
  --probe-type trust \
  --detector-method mms

# Linear Artificial Tomography (RepE)
modal run train_on_modal.py train \
  --probe-type trust \
  --detector-method lat
```

### Smaller Batch Size (if OOM)

```bash
modal run train_on_modal.py train \
  --probe-type trust \
  --batch-size 2
```

## Expected Output

```
Loading game data from results/experiment_treatment...
Loaded 10 games

Training trust probe on Modal...
  Detector method: lr
  Layers: [22]
  Batch size: 4

[Modal execution starts]
Extracted 155 interactions
Created 30 training samples
  Positive: 15
  Negative: 15
Loading model llama-70b-3.3...
Tokenizing...
Extracting activations...
  Positive samples: 15
  Negative samples: 15
Training lr detector...

Training results:
  Positive mean: 2.450
  Negative mean: -2.310
  Separation: 4.760

Detector saved to volume: /data/trained_probes/trust_lr_l22
Detector size: 131072 bytes

================================================================================
TRAINING COMPLETE
================================================================================

✓ Detector saved locally to: ./probes/trust_lr_l22/detector.pt

{
  "probe_type": "trust",
  "detector_method": "lr",
  "layers": [22],
  "num_games": 10,
  "num_interactions": 155,
  "num_positive": 15,
  "num_negative": 15,
  "positive_mean_score": 2.450,
  "negative_mean_score": -2.310,
  "separation": 4.760,
  "local_path": "./probes/trust_lr_l22/detector.pt"
}
```

## Troubleshooting

**"modal command not found"**
```bash
pip install modal
modal setup  # Follow authentication steps
```

**"No game data found"**
- Check the path to your results directory
- Ensure it contains `game*/game_stats.json` files

**"Not enough data"**
- You need games with `player_activations` in `game_stats.json`
- Run more games with `modal_probe` backend enabled

**"Out of memory"**
- Reduce batch size: `--batch-size 2`
- This is rare with 4x H100 (320GB total GPU memory)

## Costs

Modal pricing (approximate):
- 4x H100 GPUs: ~$10/hour
- Expected training time: 10-15 minutes
- **Cost per probe: ~$2-3**

The cost is low because:
1. Model loads once
2. Training is fast (< 1 minute)
3. Most time is activation extraction

## Next Steps

After training:
1. Load probe with `ApolloProbeWrapper(detector_path="./probes/trust_lr_l22/detector.pt")`
2. Test on held-out games
3. Integrate into gameplay
4. Train more probes with different configurations
