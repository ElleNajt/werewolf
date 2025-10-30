# Setup and Run Trust Probe Training

## ✅ Modal is Installed

Modal has been installed to `probe_venv/`

## Step 1: Authenticate with Modal (First Time Only)

Run this command interactively:

```bash
probe_venv/bin/modal setup
```

This will:
1. Open a browser window
2. Ask you to sign in to Modal
3. Save authentication credentials

**Note:** This requires interactive terminal access. If you're in a non-interactive environment, you need to:
- Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment variables, OR
- Run `modal setup` from your local machine and copy `~/.modal.toml` to this container

## Step 2: Run Training

Once authenticated, run:

```bash
cd src/werewolf/probe_training
./run_training.sh
```

Or manually:

```bash
probe_venv/bin/modal run train_on_modal.py train \
  --results-dir results/experiment_treatment \
  --probe-type trust
```

## What Will Happen

1. **Local:** Loads 10 games from `results/experiment_treatment/`
2. **Modal:** Spins up 4x H100 GPUs  
3. **Modal:** Loads Llama 70B
4. **Modal:** Filters to werewolf interactions only (~30 samples)
5. **Modal:** Extracts activations from layer 22
6. **Modal:** Trains logistic regression probe
7. **Modal:** Returns base64-encoded detector
8. **Local:** Saves to `./probes/trust_lr_l22/detector.pt`

**Expected time:** 10-15 minutes  
**Expected cost:** ~$2-3

## Alternative: Check Authentication Status

```bash
probe_venv/bin/modal profile current
```

If you see a profile name, you're authenticated. If not, run `modal setup`.

## Troubleshooting

**"Not authenticated"**
```bash
probe_venv/bin/modal setup
```

**"Can't open browser"**
- Get a Modal token from https://modal.com/settings
- Set environment variables:
  ```bash
  export MODAL_TOKEN_ID="your-token-id"
  export MODAL_TOKEN_SECRET="your-token-secret"
  ```

**"Function not found"**
- The modal app needs to be deployed first
- Try: `probe_venv/bin/modal deploy train_on_modal.py`

## Manual Training (Without Modal)

If you can't use Modal, you can run training locally with a GPU (requires 70GB+ VRAM):

```bash
cd src/werewolf/probe_training

# In an environment with PyTorch + transformers
python3 train_trust_probe.py \
  --results-dir results/experiment_treatment \
  --output-dir probes/trust_lr_l22 \
  --probe-type trust \
  --layers 22
```

This requires:
- GPU with 70GB+ memory (4x A100 or similar)
- PyTorch, transformers, accelerate installed
- The deception-detection repository cloned

## Files Created

After successful training:

```
src/werewolf/probe_training/probes/
└── trust_lr_l22/
    └── detector.pt          # ← Your trained probe!
```

You can then load this with:

```python
from apollo_probe import ApolloProbeWrapper

probe = ApolloProbeWrapper(
    detector_path="./probes/trust_lr_l22/detector.pt",
    layers=[22]
)
```
