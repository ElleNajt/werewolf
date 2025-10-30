# Multi-GPU Setup for Apollo Probes

## How It Works

The code **automatically distributes** Llama 70B across all available GPUs using `device_map="auto"`.

### What `device_map="auto"` Does

When you load the model, HuggingFace's `accelerate` library:

1. **Detects all available GPUs**
2. **Measures available VRAM** on each GPU
3. **Estimates memory needed** for each layer
4. **Distributes layers** across GPUs to balance memory
5. **Handles tensor movement** automatically during forward pass

Example distribution on 4x A6000 (48GB each):
```
GPU 0: Layers 0-19   (~35GB)
GPU 1: Layers 20-39  (~35GB)
GPU 2: Layers 40-59  (~35GB)
GPU 3: Layers 60-79  (~35GB)
Total: 80 layers across 4 GPUs
```

## Zero Configuration Required

**You don't need to do anything!** Just:

1. Have multiple GPUs visible to CUDA
2. Run your code normally
3. The model will automatically distribute

## Verifying Multi-GPU Setup

### Quick Check (30 seconds)
```bash
cd /workspace/src/werewolf
python test_multi_gpu.py --skip-generation
```

This shows:
- How many GPUs are detected
- Total VRAM available
- Whether it's sufficient for FP16/8-bit/4-bit

### Full Test (~5-10 minutes)
```bash
python test_multi_gpu.py
```

This also:
- Loads the model
- Shows how layers are distributed
- Tests generation
- Tests probe scoring
- Monitors memory usage

## Expected Output

### 4x A6000 48GB (192GB total)
```
✅ Found 4 GPU(s)
   GPU 0: NVIDIA RTX A6000, 48.0 GB VRAM
   GPU 1: NVIDIA RTX A6000, 48.0 GB VRAM
   GPU 2: NVIDIA RTX A6000, 48.0 GB VRAM
   GPU 3: NVIDIA RTX A6000, 48.0 GB VRAM

   Total VRAM: 192.0 GB
   ✅ Sufficient for FP16 (need ~200-240GB)

Parameter distribution:
   cuda:0: 17.50B parameters
   cuda:1: 17.50B parameters
   cuda:2: 17.50B parameters
   cuda:3: 17.50B parameters

   Total: 70.00B parameters
   ✅ Model distributed across 4 devices
```

### Single A100 80GB (insufficient for FP16)
```
✅ Found 1 GPU(s)
   GPU 0: NVIDIA A100-SXM4-80GB, 80.0 GB VRAM

   Total VRAM: 80.0 GB
   ⚠️  Only sufficient for 4-bit (need ~80-105GB)

Parameter distribution:
   cuda:0: 70.00B parameters

   Total: 70.00B parameters
   ⚠️  Model on single device (may OOM during inference)
```

## Troubleshooting

### Problem: Only 1 GPU detected but you have multiple

**Check GPU visibility:**
```bash
nvidia-smi
```

All GPUs should appear. If not:
```bash
# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# If it's set to single GPU, unset it:
unset CUDA_VISIBLE_DEVICES
```

### Problem: OOM (Out of Memory) error during loading

**You don't have enough VRAM.** Options:

1. **Use quantization:**
   ```json
   {
     "backend_type": "probe",
     "use_quantized": true,
     "quantization_bits": 4
   }
   ```

2. **Use CPU offload** (very slow):
   Edit `agent_backend.py` line 180 to add:
   ```python
   device_map="auto",
   offload_folder="offload",  # Add this
   ```

3. **Get more GPUs** (recommended)

### Problem: OOM during generation/inference

**Peak memory exceeds VRAM.** This happens during forward pass.

**Solutions:**
- Use smaller batch size (already 1 for werewolf)
- Use gradient checkpointing (not applicable for inference)
- **Add more GPUs** (only real solution)

### Problem: Model loaded but generation is very slow

**Check if model is CPU-offloaded:**
```python
# In test_multi_gpu.py output, look for:
Parameter distribution:
   cpu: X.XXB parameters  # ← BAD! Means CPU offload
```

If you see `cpu` in the device list, you don't have enough GPU memory.

## Performance Notes

### Multi-GPU Overhead

Multi-GPU adds some overhead for tensor movement:
- **2 GPUs**: ~5-10% slower than single GPU (if it fit)
- **4 GPUs**: ~10-15% slower
- **8 GPUs**: ~15-25% slower

But this is **infinitely faster than OOM**! 🎉

### Optimal GPU Configurations

| Config | GPUs | VRAM | Speed | Cost/hr |
|--------|------|------|-------|---------|
| **Best** | 4x A6000 | 192GB | Fast | ~$2.50 |
| Good | 3x A100 80GB | 240GB | Fast | ~$6-9 |
| Okay | 2x A100 80GB | 160GB | Medium | ~$4-6 |
| Minimal | 1x A100 80GB + 4bit | 80GB | Slow | ~$2-3 |

## RunPod Multi-GPU Setup

When renting on RunPod:

1. **Filter by GPU count**
   - Select "Number of GPUs" = 4
   - Select GPU type (e.g., RTX A6000)

2. **Verify in pod**
   ```bash
   nvidia-smi --query-gpu=name,memory.total --format=csv
   ```

3. **Run test**
   ```bash
   cd /workspace/src/werewolf
   python test_multi_gpu.py
   ```

4. **If successful, run game**
   ```bash
   python game_coordinator.py config.apollo_probe.json
   ```

## Key Takeaway

✅ **Multi-GPU works automatically with zero configuration**

Just make sure:
- All GPUs are visible to CUDA (`nvidia-smi`)
- You have enough total VRAM (~200GB for FP16)
- You run the test script to verify before a long game

The code handles everything else!
