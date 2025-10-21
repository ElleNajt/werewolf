# GPU Requirements for Werewolf with Apollo Probes

## TL;DR

⚠️ **IMPORTANT**: The 2-3x memory rule applies! You need significantly more VRAM than just the model size.

**Recommended setup:**
- **GPU**: 2x A100 80GB or 4x A6000 48GB (160-192GB total)
- **Config**: FP16 (no quantization)
- **Memory**: ~140GB model + ~60-80GB activations/KV = **~200-220GB peak**

**Budget option (tight fit):**
- **GPU**: Single A100 80GB
- **Config**: 4-bit quantization
- **Memory**: ~35GB model + ~40-50GB activations/KV = **~75-85GB peak**
- ⚠️ **Warning**: Significant probe accuracy degradation

## Architecture Change

**OLD (inefficient):**
```
vLLM server (Llama 70B) → Generate responses
     +
Apollo model (Llama 70B) → Extract activations
     = TWO copies of model in memory (~280GB)
```

**NEW (efficient):**
```
ProbeBackend (Llama 70B) → Generate responses AND extract activations
     = ONE copy of model in memory (~140GB or less with quantization)
```

## Memory Requirements by Configuration

### 1. Full Precision (FP16) - Recommended
```json
{
  "backend_type": "probe",
  "use_quantized": false
}
```

**Memory:**
- Model weights: 70B params × 2 bytes = **140GB**
- Activations during forward pass: ~30-50GB
- KV cache for generation: ~20-30GB
- Overhead/fragmentation: ~10-20GB
- **Peak usage: 200-240GB** (using 2.5-3x rule)

**Required GPUs:**
- ❌ A100 80GB (single) - **NOT ENOUGH!**
- ❌ H100 80GB (single) - **NOT ENOUGH!**
- ✅ 2x A100 80GB (160GB total, **tight fit**)
- ✅ 2x H100 80GB (160GB total, tight fit)
- ✅ 4x A6000 48GB (192GB total, **recommended**)
- ✅ 3x A100 80GB (240GB total, plenty of room)

**Pros:**
- Best probe accuracy (trained on FP16)
- No quantization overhead
- Faster inference

**Cons:**
- Requires expensive 80GB+ GPUs

### 2. 8-bit Quantization
```json
{
  "backend_type": "probe",
  "use_quantized": true,
  "quantization_bits": 8
}
```

**Memory:**
- Model weights: 70B params × 1 byte = **70GB**
- Activations during forward pass: ~30-40GB
- KV cache for generation: ~20-30GB
- Overhead/fragmentation: ~10-15GB
- **Peak usage: 130-155GB** (using ~2x rule)

**Required GPUs:**
- ❌ A100 80GB (single) - **NOT ENOUGH!**
- ❌ A100 40GB (single) - **NOT ENOUGH!**
- ⚠️ 2x A100 40GB (80GB total) - **Very tight, may OOM**
- ✅ 2x A6000 48GB (96GB total, **tight fit**)
- ✅ 2x A100 80GB (160GB total, plenty of room)

**Pros:**
- Fits on cheaper GPUs
- Still reasonable quality
- 50% memory savings

**Cons:**
- Slightly degraded probe accuracy
- Slower than FP16 (quantization overhead)
- **Requires bitsandbytes library**

### 3. 4-bit Quantization - Only Option for Single 80GB GPU
```json
{
  "backend_type": "probe",
  "use_quantized": true,
  "quantization_bits": 4
}
```

**Memory:**
- Model weights: 70B params × 0.5 bytes = **35GB**
- Activations during forward pass: ~25-35GB
- KV cache for generation: ~15-25GB
- Overhead/fragmentation: ~5-10GB
- **Peak usage: 80-105GB** (using ~2x rule)

**Required GPUs:**
- ⚠️ A100 80GB (single) - **Very tight fit, may work**
- ⚠️ A6000 48GB (single) - **NOT ENOUGH!**
- ✅ 2x A6000 48GB (96GB total, comfortable)
- ✅ 2x A100 40GB (80GB total, tight)

**Pros:**
- Fits on single cheaper GPU
- 75% memory savings

**Cons:**
- **Significant quality degradation**
- Probe may be less accurate (trained on FP16)
- Slowest inference
- **Requires bitsandbytes library**
- **Not recommended unless necessary**

## Probe Accuracy Considerations

The Apollo probes were trained on **FP16 activations**. Using quantization:
- **8-bit**: Minor impact (~5-10% accuracy loss), activations still close to FP16
- **4-bit**: Significant impact (~15-30% accuracy loss), activations may differ substantially

**Recommendation**:
- **Best**: FP16 on multi-GPU setup (4x A6000 or 2-3x A100 80GB)
- **Acceptable**: 4-bit on single A100 80GB if budget constrained
- **Avoid**: Single GPU without quantization - it won't fit!

## RunPod GPU Options

| GPU Setup | Total VRAM | Hourly Cost | Recommendation |
|-----------|-----------|-------------|----------------|
| 4x A6000 48GB | 192GB | ~$2.50/hr | ✅ **Best for FP16** |
| 3x A100 80GB | 240GB | ~$6-9/hr | ✅ FP16, lots of room |
| 2x A100 80GB | 160GB | ~$4-6/hr | ⚠️ FP16, tight fit |
| 2x H100 80GB | 160GB | ~$6-8/hr | ⚠️ FP16, tight (faster) |
| 1x A100 80GB | 80GB | ~$2-3/hr | ⚠️ **4-bit only** |
| 1x H100 80GB | 80GB | ~$3-4/hr | ⚠️ **4-bit only** |
| 2x A6000 48GB | 96GB | ~$1.50/hr | ❌ Too small even for 8-bit |

**Cost-performance winner**: **4x A6000 48GB** - Best balance of price and FP16 capability

## Testing Memory Usage

Test the memory footprint before running a full game:

```bash
# Test model loading (doesn't run game)
python -c "
from werewolf.agent_backend import create_backend
backend = create_backend('probe', use_quantized=False)
backend._ensure_loaded()
print('Model loaded successfully')
"

# Monitor GPU memory
watch -n 1 nvidia-smi
```

## Installation for Quantization

If using quantization, install bitsandbytes:

```bash
pip install bitsandbytes accelerate
```

## Recommended Config Per GPU Setup

**4x A6000 48GB or 3x A100 80GB (192-240GB total):**
```json
{
  "backend_type": "probe",
  "use_quantized": false
}
```
✅ Best option - FP16 with room to spare

**2x A100 80GB (160GB total):**
```json
{
  "backend_type": "probe",
  "use_quantized": false
}
```
⚠️ Tight fit for FP16 - monitor memory!

**Single A100 80GB:**
```json
{
  "backend_type": "probe",
  "use_quantized": true,
  "quantization_bits": 4
}
```
⚠️ Only option - significant accuracy loss

**Anything smaller:**
❌ Not recommended - won't fit reliably

## Performance vs Memory Tradeoff

| Config | Peak Memory | Min GPU Setup | Speed | Probe Quality |
|--------|-------------|---------------|-------|---------------|
| FP16 | 200-240GB | 4x A6000 or 2-3x A100 | Fast | ⭐⭐⭐⭐⭐ Best |
| 8-bit | 130-155GB | 2x A100 80GB | Medium | ⭐⭐⭐⭐ Good |
| 4-bit | 80-105GB | 1x A100 80GB | Slow | ⭐⭐ Poor |

**Reality check**: Single 80GB GPUs are NOT sufficient for FP16 or 8-bit!

## Multi-GPU Setup

For model parallelism across multiple GPUs, the Apollo code uses `device_map="auto"` which automatically distributes:

```python
# This happens automatically in the backend
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Distributes across available GPUs
    torch_dtype=torch.float16
)
```

No additional configuration needed - it just works!
