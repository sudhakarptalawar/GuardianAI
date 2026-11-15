"""
GuardianAI — Main Entry Point
Run the unified detection pipeline + API server together.

Usage:
    python main.py                           # webcam, all detectors, live preview
    python main.py --source videos/fire/fire.mp4
    python main.py --source rtsp://...
    python main.py --detectors fire person   # select specific detectors
    python main.py --no-show                 # headless (no preview window)
    python main.py --no-api                  # pipeline only, no API server
    python main.py --api-only                # API server only, no camera
"""

import argparse
import threading
import uvicorn

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
    p = argparse.ArgumentParser(description="GuardianAI — Intelligent Safety System")
    p.add_argument("--source",     default=0,           help="Camera index or RTSP/file URL (default: 0)")
    p.add_argument("--camera-id",  default="cam_0",     help="Logical camera ID")
    p.add_argument("--name",       default="Camera 0",  help="Camera display name")
    p.add_argument("--location",   default="Unknown",   help="Camera location label")
    p.add_argument("--fps",        type=int, default=5, help="Target processing FPS (default: 5)")
    p.add_argument(
        "--detectors",
        nargs="+",
        default=["person", "fire", "snake", "object"],
        choices=list(DETECTOR_MAP.keys()),
        help="Detectors to enable (default: all)",
    )
    p.add_argument("--no-show",  action="store_true", help="Disable live preview window")
    p.add_argument("--no-api",   action="store_true", help="Disable FastAPI server")
    p.add_argument("--api-only", action="store_true", help="Run API server only (no camera)")
    p.add_argument("--api-port", type=int, default=8000, help="API server port (default: 8000)")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (testing)")
    return p.parse_args()


def start_api(port: int):
    """Start the FastAPI/uvicorn server in the current thread."""
    uvicorn.run(
        "api.app:app",
        host    = "0.0.0.0",
        port    = port,
        reload  = False,
        log_level = "info",
    )


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("\n" + "=" * 55)
    print("   🛡️  GuardianAI — Intelligent Safety System v1.0")
    print("=" * 55)

    # ── API-only mode ─────────────────────────────────────────────────────────
    if args.api_only:
        print(f"   Mode    : API server only")
        print(f"   API     : http://localhost:{args.api_port}/docs")
        print("=" * 55 + "\n")
        start_api(args.api_port)
        return

    # ── Normal mode (pipeline ± API) ──────────────────────────────────────────
    print(f"   Camera  : {args.name} ({args.camera_id})")
    print(f"   Source  : {source}")
    print(f"   Location: {args.location}")
    print(f"   FPS     : {args.fps}")
    print(f"   Detectors: {', '.join(args.detectors)}")
    if not args.no_api:
        print(f"   API     : http://localhost:{args.api_port}/docs")
    print("=" * 55 + "\n")

    # Shared event manager
    event_manager = EventManager()

    # Inject into API routes
    from api.routes.events import set_event_manager
    set_event_manager(event_manager)

    # Start API in background thread
    if not args.no_api:
        api_thread = threading.Thread(
            target=start_api,
            args=(args.api_port,),
            daemon=True,
        )
        api_thread.start()
        print(f"[Main] ✅ API server started on port {args.api_port}\n")

    # Build pipeline
    pipeline = Pipeline(
        camera_id     = args.camera_id,
        source        = source,
        name          = args.name,
        location      = args.location,
        fps_target    = args.fps,
        event_manager = event_manager,
    )

    # Register detectors
    for detector_name in args.detectors:
        detector = DETECTOR_MAP[detector_name]()
        pipeline.add_detector(detector.detect)
        print(f"[Main] ✅ Registered detector: {detector_name}")

    print()

    # Run pipeline (blocking)
    pipeline.run(
        show     = not args.no_show,
        duration = args.duration,
    )


if __name__ == "__main__":
    main()
