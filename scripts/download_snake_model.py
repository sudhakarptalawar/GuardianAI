"""
GuardianAI – One-time script to download the snake detection model from Roboflow.

Usage:
  python scripts/download_snake_model.py --api-key YOUR_KEY

Get a free API key at: https://app.roboflow.com/settings/api
The model is saved to models/snake.pt and auto-detected by detectors/snake.py.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download snake.pt from Roboflow")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--workspace", default="amir-pdcee", help="Roboflow workspace")
    parser.add_argument("--project",   default="snake-detection-gat5j", help="Roboflow project")
    parser.add_argument("--version",   default=2, type=int, help="Model version")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    output = "models/snake.pt"

    print(f"Downloading snake model from Roboflow ({args.workspace}/{args.project} v{args.version})...")
    try:
        from roboflow import Roboflow
        rf      = Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project)
        version = project.version(args.version)
        # Download YOLOv8 format model
        version.download("yolov8", location="models/snake_rf_tmp")
        # Find the .pt file
        import glob
        pts = glob.glob("models/snake_rf_tmp/**/*.pt", recursive=True)
        if pts:
            import shutil
            shutil.copy(pts[0], output)
            print(f"✅ Saved to {output}")
        else:
            print("❌ No .pt file found in downloaded package. Try exporting manually from Roboflow.")
            sys.exit(1)
    except ImportError:
        print("❌ roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
