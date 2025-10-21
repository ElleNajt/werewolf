# Architecture Change: Unified Model for Generation + Probing

## What Changed

I've **redesigned the ProbeBackend** to eliminate the memory inefficiency from the original design.

### Before (Your Original Design)
```
┌─────────────────────────────────┐
│  vLLM Server (Llama 70B)        │  ← Generate game responses
│  Memory: ~140GB                 │
└─────────────────────────────────┘
              +
┌─────────────────────────────────┐
│  Apollo Probe (Llama 70B)       │  ← Extract activations for scoring
│  Memory: ~140GB                 │
└─────────────────────────────────┘
              =
         **~280GB total** ❌
```

This would require TWO copies of Llama 70B in memory!

### After (New Design)
```
┌─────────────────────────────────┐
│  ProbeBackend (Llama 70B)       │
│  - Generate responses           │  ← Single model does both
│  - Extract activations          │
│  - Score with probe             │
│  Memory: ~140GB (or less)       │
└─────────────────────────────────┘
              =
         **~140GB total** ✅
```

One model does everything!

## Memory Requirements (Corrected with 2-3x Rule)

⚠️ **The 2-3x memory rule applies!** You need VRAM for model + activations + KV cache + overhead.

| Config | Model | Peak Usage | Min GPU Setup |
|--------|-------|------------|---------------|
| **FP16** (best) | 140GB | **200-240GB** | 4x A6000 (192GB) or 2-3x A100 80GB |
| **8-bit** | 70GB | **130-155GB** | 2x A100 80GB (160GB) |
| **4-bit** (only for single 80GB) | 35GB | **80-105GB** | 1x A100 80GB |

## Key Benefits

1. **50% memory savings** - Only one model in memory
2. **Simpler deployment** - No separate servers needed
3. **Same accuracy** - Still uses Apollo's pre-trained probes
4. **Quantization support** - Can reduce memory further if needed

## How to Use

### FP16 (No Quantization) - Best Quality
```json
{
  "backend_type": "probe",
  "use_quantized": false
}
```

**Requires: 4x A6000 48GB (192GB) or 2-3x A100 80GB**

✅ Best probe accuracy

### 4-bit Quantization - Only Option for Single 80GB GPU
```json
{
  "backend_type": "probe",
  "use_quantized": true,
  "quantization_bits": 4
}
```

**Requires: 1x A100 80GB**

⚠️ **Warning**: 15-30% accuracy degradation on probe scores!

## What Stays the Same

- Still uses Apollo's pre-trained probes
- Same probe scoring method (layer 22 activations)
- Same game mechanics
- Same accuracy for FP16 mode

## Bottom Line

**Reality Check**: Single 80GB GPUs are **NOT** sufficient for FP16!

**Best Option**: **4x A6000 48GB** (~$2.50/hr) with **FP16** for full accuracy
**Budget Option**: **1x A100 80GB** (~$2-3/hr) with **4-bit** (degraded accuracy)
**Don't Do**: Single GPU with FP16 or 8-bit - will OOM!

See [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) for detailed memory analysis.

## Apollo Code Reuse

We're using Apollo's infrastructure directly:
- ✅ `Activations.from_model()` for extracting hidden states
- ✅ `LogisticRegressionDetector` for probe scoring
- ✅ `get_llama3_model_and_tokenizer()` for model loading
- ✅ `TokenizedDataset` for tokenization
- ✅ Layer 22 activations (same as Apollo's roleplaying probe)

The only addition is `model.generate()` for response generation, which Apollo doesn't have (they only do probe inference on pre-generated text).
