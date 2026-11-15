"""
GuardianAI — Fire & Smoke Detector
Uses the custom-trained fire.pt model for high-accuracy fire/smoke detection.
Falls back to yolov8n.pt if fire.pt is not found.
"""

import os
from typing import List
from ultralytics import YOLO

from core.severity import assess, ThreatAssessment


class FireDetector:
    """
    Detects fire and smoke in a frame using a custom-trained YOLO model.
    """

    PRIMARY_MODEL  = "models/fire.pt"
    FALLBACK_MODEL = "yolov8n.pt"
    CONF_THRESHOLD = 0.35

    def __init__(self, model_path: str | None = None):
        path = model_path or (
            self.PRIMARY_MODEL if os.path.exists(self.PRIMARY_MODEL)
            else self.FALLBACK_MODEL
        )
        print(f"[FireDetector] Loading model: {path}")
        self.model = YOLO(path)
        print("[FireDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Run detection on a single frame and return assessments."""
        results     = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)
        assessments = []

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                obj  = result.names[cls]
                conf = float(box.conf[0])

                if obj.lower() in ("fire", "smoke"):
                    assessments.append(assess(obj=obj, confidence=conf))

        return assessments
