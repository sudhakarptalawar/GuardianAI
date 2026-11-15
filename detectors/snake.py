"""
GuardianAI — Snake Detector
Detects snakes in a frame using YOLOv8n.
"""

from typing import List
from ultralytics import YOLO

from core.severity import assess, ThreatAssessment


class SnakeDetector:
    """
    Detects snakes in a frame using YOLOv8n.
    """

    MODEL_PATH     = "yolov8n.pt"
    CONF_THRESHOLD = 0.35

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[SnakeDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[SnakeDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Run detection on a single frame and return assessments."""
        results     = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)
        assessments = []

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                obj  = result.names[cls]
                conf = float(box.conf[0])

                if obj.lower() == "snake":
                    assessments.append(assess(obj=obj, confidence=conf))

        return assessments
