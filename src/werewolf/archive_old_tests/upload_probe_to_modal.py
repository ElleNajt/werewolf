#!/usr/bin/env python3
"""
Upload detector probe to Modal volume.

Usage:
    python upload_probe_to_modal.py lie_detector_8b_layer12.pt
"""

import modal
import sys
from pathlib import Path

# Connect to the volume
volume = modal.Volume.from_name("werewolf-apollo-probes")

def upload_detector(local_path: str):
    """Upload detector to Modal volume."""
    local_file = Path(local_path)
    
    if not local_file.exists():
        print(f"Error: File not found: {local_path}")
        sys.exit(1)
    
    # Determine remote path based on filename
    filename = local_file.name
    remote_path = f"/detectors/{filename}"
    
    print(f"Uploading {local_path} to Modal volume...")
    print(f"  Volume: werewolf-apollo-probes")
    print(f"  Remote path: {remote_path}")
    
    # Upload file using batch upload
    volume.add_local_file(local_file, remote_path)
    volume.commit()
    
    print(f"✓ Upload complete!")
    print(f"\nTo use this detector in your config:")
    print(f'  "detector_path": "/models{remote_path}"')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_probe_to_modal.py <detector_file.pt>")
        sys.exit(1)
    
    upload_detector(sys.argv[1])
