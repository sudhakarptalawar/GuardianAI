"""
GuardianAI — Main Entry Point
Run the unified detection pipeline from the command line.

Usage:
    python main.py                          # webcam, all detectors, live preview
    python main.py --source videos/fire/fire.mp4
    python main.py --source rtsp://...
    python main.py --detectors fire person  # select specific detectors
    python main.py --no-show                # headless (no preview window)
"""

import argparse

from core.pipeline      import Pipeline
from core.event_manager import EventManager
from detectors.person   import PersonDetector
from detectors.fire     import FireDetector
from detectors.snake    import SnakeDetector
from detectors.objects  import ObjectDetector


DETECTOR_MAP = {
    "person": PersonDetector,
    "fire":   FireDetector,
    "snake":  SnakeDetector,
    "object": ObjectDetector,
}


def parse_args():
    p = argparse.ArgumentParser(description="GuardianAI — Unified Detection Pipeline")
    p.add_argument("--source",    default=0,          help="Camera index or RTSP/file URL (default: 0)")
    p.add_argument("--camera-id", default="cam_0",    help="Logical camera ID (default: cam_0)")
    p.add_argument("--name",      default="Camera 0", help="Camera display name")
    p.add_argument("--location",  default="Unknown",  help="Camera location label")
    p.add_argument("--fps",       type=int, default=5, help="Target processing FPS (default: 5)")
    p.add_argument(
        "--detectors",
        nargs="+",
        default=["person", "fire", "snake", "object"],
        choices=list(DETECTOR_MAP.keys()),
        help="Detectors to enable (default: all)",
    )
    p.add_argument("--no-show", action="store_true", help="Disable live preview window")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (testing)")
    return p.parse_args()


def main():
    args = parse_args()

    # Convert source to int if it's a digit string (e.g. "0" → 0)
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("\n" + "=" * 50)
    print("   🛡️  GuardianAI — Intelligent Safety System")
    print("=" * 50)
    print(f"   Camera  : {args.name} ({args.camera_id})")
    print(f"   Source  : {source}")
    print(f"   Location: {args.location}")
    print(f"   FPS     : {args.fps}")
    print(f"   Detectors: {', '.join(args.detectors)}")
    print("=" * 50 + "\n")

    # Build shared event manager
    event_manager = EventManager()

    # Build pipeline
    pipeline = Pipeline(
        camera_id     = args.camera_id,
        source        = source,
        name          = args.name,
        location      = args.location,
        fps_target    = args.fps,
        event_manager = event_manager,
    )

    # Register selected detectors
    for detector_name in args.detectors:
        detector_cls = DETECTOR_MAP[detector_name]
        detector     = detector_cls()
        pipeline.add_detector(detector.detect)
        print(f"   ✅ Registered detector: {detector_name}")

    print()

    # Run
    pipeline.run(
        show     = not args.no_show,
        duration = args.duration,
    )


if __name__ == "__main__":
    main()
