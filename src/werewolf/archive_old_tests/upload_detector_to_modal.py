"""
Upload a detector file to the Modal volume for use with werewolf game.
"""
import modal
from pathlib import Path

# Use the same volume as the backend
VOLUME = modal.Volume.from_name("werewolf-apollo-probes", create_if_missing=True)
VOLUME_PATH = "/models"

app = modal.App("upload-detector")

@app.function(volumes={VOLUME_PATH: VOLUME})
def upload_detector(detector_data: bytes, remote_filename: str):
    """
    Upload a detector file to the Modal volume.
    
    Args:
        detector_data: Detector file data as bytes
        remote_filename: Filename to save as on Modal volume (e.g., 'lie_detector_70b_layer12.pt')
    """
    from pathlib import Path
    
    # Ensure detectors directory exists
    detector_dir = Path(VOLUME_PATH) / "detectors"
    detector_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to Modal volume
    remote_path = detector_dir / remote_filename
    with open(remote_path, 'wb') as f:
        f.write(detector_data)
    
    # Commit the volume changes
    VOLUME.commit()
    
    print(f"✓ Uploaded {len(detector_data)} bytes to {remote_path}")
    print(f"✓ Volume committed")
    
    return str(remote_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # Single file upload
        local_path = sys.argv[1]
        remote_filename = sys.argv[2]
        
        # Read file locally
        with open(local_path, 'rb') as f:
            detector_data = f.read()
        
        with app.run():
            result = upload_detector.remote(detector_data, remote_filename)
            print(f"Detector available at: {result}")
    
    elif len(sys.argv) == 2 and sys.argv[1] == "--upload-all":
        # Upload all detectors from deception-detection directory
        from pathlib import Path
        
        detector_base = Path("../../deception-detection/example_results")
        detectors = [
            (detector_base / "instructed_pairs/detector.pt", "lie_detector_70b_instructed_pairs.pt"),
            (detector_base / "roleplaying/detector.pt", "lie_detector_70b_roleplaying.pt"),
            (detector_base / "descriptive/detector.pt", "lie_detector_70b_descriptive.pt"),
            (detector_base / "followup/detector.pt", "lie_detector_70b_followup.pt"),
            (detector_base / "sae_rp/detector.pt", "lie_detector_70b_sae_rp.pt"),
        ]
        
        with app.run():
            for local_path, remote_filename in detectors:
                if local_path.exists():
                    print(f"\nUploading {local_path.name}...")
                    # Read file locally
                    with open(local_path, 'rb') as f:
                        detector_data = f.read()
                    result = upload_detector.remote(detector_data, remote_filename)
                    print(f"✓ {remote_filename}")
                else:
                    print(f"✗ Skipping {local_path} (not found)")
        
        print("\n✓ All detectors uploaded!")
    
    else:
        print("Usage:")
        print("  python upload_detector_to_modal.py <local_path> <remote_filename>")
        print("  python upload_detector_to_modal.py --upload-all")
        print("\nExamples:")
        print("  python upload_detector_to_modal.py detector.pt lie_detector_70b.pt")
        print("  python upload_detector_to_modal.py --upload-all")
        sys.exit(1)
