"""
GuardianAI — Person Detector
Detects people using YOLOv8n and returns ThreatAssessments via the severity engine.
"""

from typing import List
from ultralytics import YOLO

from core.severity import assess, ThreatAssessment


class PersonDetector:
    """
    Detects people in a frame using YOLOv8n.
    Uses time-of-day context to escalate severity at night.
    """

    MODEL_PATH = "yolov8n.pt"
    CONF_THRESHOLD = 0.4

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[PersonDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[PersonDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Run detection on a single frame and return assessments."""
        results      = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)
        assessments  = []
        person_count = 0

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
                    assessments.append(assessment)

        return assessments
