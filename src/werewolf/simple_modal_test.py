#!/usr/bin/env python
"""
Simple Modal connection test that bypasses environment issues.
"""

import sys

try:
    import modal
    print("✓ Modal imported successfully")
    print(f"  Modal version: {modal.__version__}")
except ImportError as e:
    print(f"✗ Failed to import modal: {e}")
    print("\nPlease install modal:")
    print("  pip install modal")
    sys.exit(1)

# Try to connect to the Modal app
try:
    print("\n1. Connecting to Modal app...")
    cls = modal.Cls.from_name("werewolf-apollo-probe", "ApolloProbeService")
    service = cls()
    print("✓ Connected to Modal service!")

    # Try a simple generation test
    print("\n2. Testing generation...")
    result = service.generate_with_probe.remote(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' in JSON format: {\"message\": \"...\"}"}
        ],
        max_tokens=50,
        temperature=0.5
    )

    if "error" in result:
        print(f"✗ Generation failed: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
        sys.exit(1)

    print("✓ Generation successful!")
    print(f"\n  Generated text: {result.get('generated_text', 'N/A')[:100]}...")
    print(f"  Aggregate score: {result.get('aggregate_score', 'N/A')}")
    print(f"  Num tokens: {result.get('num_tokens', 'N/A')}")

    print("\n=== All tests passed! ===")
    print("\nThe Modal backend is working correctly.")
    print("You can now run the werewolf game with:")
    print("  python game_coordinator.py config.modal_probe.json")

except modal.exception.NotFoundError as e:
    print(f"✗ Modal app not found: {e}")
    print("\nDid you deploy the Modal app?")
    print("  modal deploy modal_apollo_backend.py")
    sys.exit(1)

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
