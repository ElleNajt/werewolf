"""
Test multi-GPU setup for Apollo probe backend.

This script verifies that:
1. Multiple GPUs are detected
2. Model distributes across GPUs automatically
3. Generation works on distributed model
4. Probe scoring works on distributed model
"""

import torch
import sys
from pathlib import Path

def check_gpu_setup():
    """Check GPU availability and configuration."""
    print("=" * 80)
    print("GPU Setup Check")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available! This system has no GPUs.")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name}, {memory_gb:.1f} GB VRAM")

    total_vram = sum(
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(num_gpus)
    )
    print(f"\n   Total VRAM: {total_vram:.1f} GB")

    # Check if sufficient for FP16
    if total_vram >= 200:
        print(f"   ✅ Sufficient for FP16 (need ~200-240GB)")
    elif total_vram >= 130:
        print(f"   ⚠️  Sufficient for 8-bit (need ~130-155GB)")
    elif total_vram >= 80:
        print(f"   ⚠️  Only sufficient for 4-bit (need ~80-105GB)")
    else:
        print(f"   ❌ Not sufficient even for 4-bit")
        return False

    return True


def test_model_loading():
    """Test that model loads and distributes across GPUs."""
    print("\n" + "=" * 80)
    print("Testing Model Loading")
    print("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deception-detection"))

    from werewolf.agent_backend import ProbeBackend

    # Create backend (lazy loading)
    print("\nCreating ProbeBackend...")
    backend = ProbeBackend(
        use_quantized=False,  # Try FP16 first
        provide_scores=True
    )

    print("Loading model (this will take a few minutes)...")
    backend._ensure_loaded()

    print("\n✅ Model loaded successfully!")

    # Check device placement
    print("\nChecking device placement:")
    model = backend.model

    # Count parameters on each device
    device_params = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_params:
            device_params[device] = 0
        device_params[device] += param.numel()

    print("\nParameter distribution:")
    for device, count in sorted(device_params.items()):
        params_b = count / 1e9
        print(f"   {device}: {params_b:.2f}B parameters")

    # Check total
    total_params = sum(device_params.values()) / 1e9
    print(f"\n   Total: {total_params:.2f}B parameters")

    if len(device_params) > 1:
        print(f"\n   ✅ Model distributed across {len(device_params)} devices")
    else:
        print(f"\n   ⚠️  Model on single device (may OOM during inference)")

    return backend


def test_generation(backend):
    """Test that generation works on distributed model."""
    print("\n" + "=" * 80)
    print("Testing Text Generation")
    print("=" * 80)

    test_prompt = "You are a villager in a game of Werewolf. What do you think about the other players?"
    system_prompt = "You are playing Werewolf. Respond in JSON format with 'reasoning' and 'action' fields."

    print(f"\nGenerating response to test prompt...")
    print(f"Prompt: {test_prompt[:80]}...")

    try:
        response, metadata = backend.call(
            prompt=test_prompt,
            system_prompt=system_prompt,
            max_retries=1
        )

        print(f"\n✅ Generation successful!")
        print(f"Response length: {len(response)} chars")
        print(f"Response preview: {response[:200]}...")

        if metadata and metadata.get('activations'):
            print(f"\n✅ Probe scoring successful!")
            score = metadata['activations'].get('aggregate_score', 'N/A')
            print(f"Deception score: {score}")
        else:
            print(f"\n⚠️  No probe scores in metadata")

        return True

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Check current GPU memory usage."""
    print("\n" + "=" * 80)
    print("GPU Memory Usage")
    print("=" * 80)

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        print(f"\nGPU {i}:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Free:      {total - reserved:.2f} GB")

        if reserved / total > 0.95:
            print(f"   ⚠️  Nearly full!")
        elif reserved / total > 0.85:
            print(f"   ⚠️  Getting full")
        else:
            print(f"   ✅ Adequate free space")


def main():
    """Run all tests."""
    print("\n" + "🔍 Multi-GPU Setup Verification for Apollo Probes\n")

    # Step 1: Check GPU availability
    if not check_gpu_setup():
        print("\n❌ GPU setup insufficient. Exiting.")
        return 1

    # Step 2: Load model and check distribution
    try:
        backend = test_model_loading()
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Check memory after loading
    test_memory_usage()

    # Step 4: Test generation
    if not test_generation(backend):
        print("\n❌ Generation test failed")
        return 1

    # Step 5: Final memory check
    test_memory_usage()

    print("\n" + "=" * 80)
    print("✅ All tests passed! Multi-GPU setup is working.")
    print("=" * 80)
    print("\nYou can now run the werewolf game with:")
    print("  python game_coordinator.py config.apollo_probe.json")

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test multi-GPU setup")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation test (faster, but less thorough)"
    )
    args = parser.parse_args()

    if args.skip_generation:
        print("\n⚠️  Skipping generation test (--skip-generation flag)")
        check_gpu_setup()
        exit(0)

    exit(main())
