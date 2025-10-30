"""
Helper script to upload Apollo detector to Modal volume.

This uploads the detector.pt file from deception-detection to the Modal volume
so it can be accessed by the Modal service.
"""

import modal
from pathlib import Path
import sys

# Volume configuration (must match modal_apollo_backend.py)
VOLUME_NAME = "werewolf-apollo-probes"
VOLUME_PATH = "/models"
DETECTOR_DIR = Path(VOLUME_PATH) / "detectors"

# Get or create volume
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Create app for uploading
app = modal.App("upload-detector")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def upload_file(content: bytes, remote_path: str):
    """Upload file content to volume."""
    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_file, 'wb') as f:
        f.write(content)

    # Commit the volume from inside the container
    volume.commit()

    print(f"Uploaded to {remote_path}")
    return True


def upload_detector(local_detector_path: str):
    """Upload detector to Modal volume."""
    local_path = Path(local_detector_path)

    if not local_path.exists():
        print(f"Error: Detector not found at {local_path}")
        sys.exit(1)

    print(f"Uploading detector from {local_path}...")

    # Read local file
    with open(local_path, 'rb') as f:
        content = f.read()

    # Upload to Modal
    remote_path = str(DETECTOR_DIR / "detector.pt")

    with app.run():
        upload_file.remote(content, remote_path)

    print(f"✓ Detector uploaded successfully to {remote_path}")
    print(f"  Volume: {VOLUME_NAME}")
    print(f"  Use this path in config: {remote_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload Apollo detector to Modal")
    parser.add_argument(
        "detector_path",
        nargs="?",
        default="../deception-detection/example_results/roleplaying/detector.pt",
        help="Path to detector.pt file (default: deception-detection/example_results/roleplaying/detector.pt)"
    )

    args = parser.parse_args()
    upload_detector(args.detector_path)
