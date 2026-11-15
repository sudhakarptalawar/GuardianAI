"""
GuardianAI — General Object Detector
Detects common objects (vehicles, animals, etc.) using YOLOv8n
and routes them through the severity engine.
"""

from typing import List
from ultralytics import YOLO

from core.severity import assess, ThreatAssessment


# Objects we care about beyond person/fire/snake
TRACKED_OBJECTS = {
    "car", "truck", "motorcycle", "bus",    # vehicles
    "dog", "cat", "bear", "elephant",       # animals
}


class ObjectDetector:
    """
    General-purpose detector for vehicles and animals.
    """

    MODEL_PATH     = "yolov8n.pt"
    CONF_THRESHOLD = 0.4

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[ObjectDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[ObjectDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Run detection on a single frame and return assessments."""
        results     = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)
        assessments = []

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                obj  = result.names[cls]
                conf = float(box.conf[0])

                if obj.lower() in TRACKED_OBJECTS:
                    assessments.append(assess(obj=obj, confidence=conf))

        return assessments
