"""
GuardianAI — test.py (fixed & refactored)
Quick single-file test using the new core modules.
Runs the full detection pipeline on a video file and prints + saves a report.

Usage:
    python test.py
    python test.py --source videos/fire/fire.mp4
"""

import argparse
import os
from datetime import datetime
from ultralytics import YOLO

from core.severity      import assess, Severity
from core.event_manager import EventManager

os.makedirs("reports", exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="videos/fire/fire.mp4", help="Video or image path")
    p.add_argument("--model",  default="yolov8n.pt",           help="YOLO model path")
    return p.parse_args()


def run_test(source: str, model_path: str):
    print(f"\n[Test] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[Test] Running on: {source}\n")
    results = model(source, verbose=False)

    event_manager = EventManager()
    person_count  = 0

    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            obj  = result.names[cls]
            conf = float(box.conf[0])

            if obj.lower() == "person":
                person_count += 1

            assessment = assess(
                obj        = obj,
                confidence = conf,
                context    = {"person_count": person_count},
            )
            event_manager.add(assessment, camera_id="test_cam")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = event_manager.summary()
    print("\nGUARDIANAI REPORT")
    print("=" * 40)
    print(f"Total Events   : {summary['total_events']}")
    print(f"Critical       : {summary['by_severity']['CRITICAL']}")
    print(f"High           : {summary['by_severity']['HIGH']}")
    print(f"Medium         : {summary['by_severity']['MEDIUM']}")
    print(f"Low            : {summary['by_severity']['LOW']}")
    print("=" * 40)

    # ── Event list ───────────────────────────────────────────────────────────
    print("\nEVENTS:")
    for event in event_manager.get_all():
        print(f"  {event.label}  |  conf={event.confidence}  |  {event.timestamp}")

    # ── Save text report ─────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/report_{ts}.txt"
    with open(report_path, "w") as f:
        f.write("GUARDIANAI REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Source  : {source}\n")
        f.write(f"Model   : {model_path}\n")
        f.write(f"Run at  : {datetime.now()}\n\n")
        for event in event_manager.get_all():
            f.write(f"{event.label}  |  conf={event.confidence}  |  {event.timestamp}\n")
        f.write("\nSUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Events: {summary['total_events']}\n")
        for sev, count in summary["by_severity"].items():
            f.write(f"{sev}: {count}\n")

    print(f"\n✅ Report saved: {report_path}")


if __name__ == "__main__":
    args = parse_args()
    run_test(args.source, args.model)
