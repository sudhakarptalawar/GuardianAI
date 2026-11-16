"""
GuardianAI — Perimeter & Wall-Jump Detector
Detects unauthorized perimeter crossing and elevated climbing/wall-jump motion.
Monitors person coordinates relative to configurable perimeter zones.
"""

import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from core.severity import ThreatAssessment, Severity, EventCategory


@dataclass
class TrackedPerson:
    last_y: float
    last_h: float
    last_seen: float
    climbing_frames: int = 0


class PerimeterDetector:
    """
    Detects perimeter intrusion, fence/wall scaling, and climbing motion.
    Analyzes person vertical position, height ratio, and movement velocity.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        perimeter_top_ratio: float = 0.45,   # top 45% of frame considered wall/perimeter zone
        climb_velocity_threshold: float = 0.04, # upward shift per frame
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.perimeter_top_ratio = perimeter_top_ratio
        self.climb_velocity_threshold = climb_velocity_threshold
        self._history: Dict[int, TrackedPerson] = {}
        print("[PerimeterDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Detect person perimeter violations on raw video frame."""
        if frame is None:
            return []
        h, w = frame.shape[:2]
        results = self.model(frame, conf=0.35, verbose=False)
        person_boxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if result.names[cls].lower() == "person":
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    person_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))

        return self.detect_boxes((h, w), person_boxes)

    def detect_boxes(self, frame_shape: Tuple[int, int], person_boxes: List[Tuple[float, float, float, float, float]]) -> List[ThreatAssessment]:
        """
        Analyze person bounding boxes: (x1, y1, x2, y2, conf) normalized or pixel coordinates.
        Returns list of perimeter ThreatAssessments.
        """
        assessments: List[ThreatAssessment] = []
        h, w = frame_shape[:2]
        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        perimeter_line_y = h * self.perimeter_top_ratio

        for idx, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
            box_h = y2 - y1
            box_center_y = (y1 + y2) / 2.0

            # Check if person is in elevated perimeter/wall zone
            in_perimeter_zone = y1 < perimeter_line_y

            # Check track history for climbing motion
            prev = self._history.get(idx)
            is_climbing = False

            if prev:
                # Upward displacement indicates climbing/wall jump
                dy = (prev.last_y - box_center_y) / h
                if dy > self.climb_velocity_threshold and in_perimeter_zone:
                    prev.climbing_frames += 1
                    if prev.climbing_frames >= 2:
                        is_climbing = True
                else:
                    prev.climbing_frames = max(0, prev.climbing_frames - 1)

                prev.last_y = box_center_y
                prev.last_h = box_h
                prev.last_seen = now
            else:
                self._history[idx] = TrackedPerson(
                    last_y=box_center_y,
                    last_h=box_h,
                    last_seen=now,
                )

            if is_climbing or (in_perimeter_zone and box_h > 0.15 * h):
                label = "[HIGH] Wall-Climb / Perimeter Intrusion Detected" if is_climbing else "[HIGH] Perimeter Boundary Intrusion"
                desc = "Individual detected climbing wall/perimeter." if is_climbing else "Individual detected inside restricted perimeter boundary."
                assessments.append(ThreatAssessment(
                    category=EventCategory.INTRUDER,
                    severity=Severity.HIGH,
                    label=label,
                    confidence=float(conf),
                    description=desc,
                    timestamp=now_str,
                ))

        # Cleanup old tracks
        stale = [k for k, v in self._history.items() if now - v.last_seen > 2.0]
        for k in stale:
            del self._history[k]

        return assessments
