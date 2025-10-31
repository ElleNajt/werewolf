"""
Upload 70B lie detector probe to Modal volume for werewolf game.
"""

import modal
from pathlib import Path

# Volume and paths
VOLUME_NAME = "werewolf-apollo-probes"
DETECTOR_LOCAL_PATH = "../../deception-detection/example_results/instructed_pairs/detector.pt"
DETECTOR_REMOTE_PATH = "/models/detectors/lie_detector_70b_layer12.pt"

def upload_detector():
    """Upload the 70B detector to Modal volume."""
    
    # Get the volume
    volume = modal.Volume.from_name(VOLUME_NAME)
    
    # Check local file exists
    local_path = Path(DETECTOR_LOCAL_PATH)
    if not local_path.exists():
        raise FileNotFoundError(f"Detector not found at {local_path}")
    
    print(f"Uploading {local_path} to Modal volume '{VOLUME_NAME}'...")
    print(f"Remote path: {DETECTOR_REMOTE_PATH}")
    
    # Upload to volume using batch_upload
    with volume.batch_upload() as batch:
        batch.put_file(
            local_file=str(local_path),
            remote_path=DETECTOR_REMOTE_PATH
        )
    
    print("✓ Upload complete!")
    print(f"\nDetector is now available at: {DETECTOR_REMOTE_PATH}")
    print("You can now run 70B games with this detector.")

if __name__ == "__main__":
    upload_detector()
