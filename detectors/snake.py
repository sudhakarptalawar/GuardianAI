"""
GuardianAI – Snake Detector
Detects snakes using:
  1. Local custom YOLO model  (models/snake.pt)  — most accurate, fully offline
  2. Shape-heuristic fallback                    — offline, works immediately

To get the accurate snake.pt model:
  Run: python scripts/download_snake_model.py
  (Requires a free Roboflow API key — sign up at roboflow.com)
"""

import os
import logging
from typing import List
from datetime import datetime, timezone

import cv2
import numpy as np

from core.severity import assess, ThreatAssessment, Severity, EventCategory

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH  = "models/snake.pt"
CONF_THRESHOLD    = 0.35


# ---------------------------------------------------------------------------
# Shape-heuristic helper (always offline)
# ---------------------------------------------------------------------------

def _heuristic_snake_detect(frame) -> List[dict]:
    """
    Offline heuristic: finds elongated / coiled contours resembling snakes.
    Uses edge detection + ellipse fitting to find thin elongated shapes.
    Returns list of dicts: {bbox: (x1,y1,x2,y2), confidence: float}
    """
    detections = []
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w     = frame.shape[:2]
    min_area = (h * w) * 0.002   # at least 0.2% of frame area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if len(cnt) < 5:
            continue
        ellipse           = cv2.fitEllipse(cnt)
        (ex, ey), (MA, ma), angle = ellipse
        if ma == 0:
            continue
        aspect = MA / ma           # large value = very elongated
        fill   = area / max(1, MA * ma * 0.785)   # how well it fills the ellipse

        # Snakes: very elongated (aspect > 3) OR curled tight (medium aspect, dense fill)
        if aspect > 3.0 or (aspect > 1.8 and fill > 0.65):
            x, y, bw, bh = cv2.boundingRect(cnt)
            conf = min(0.55, 0.28 + (aspect - 3.0) * 0.05) if aspect > 3.0 else 0.32
            detections.append({"bbox": (x, y, x + bw, y + bh), "confidence": round(conf, 2)})

    return detections


# ---------------------------------------------------------------------------
# Snake Detector class
# ---------------------------------------------------------------------------

class SnakeDetector:
    """
    Multi-strategy snake detector.

    Strategy 1: Local YOLO model (models/snake.pt) — most accurate
    Strategy 2: Shape-heuristic contour analysis   — always works offline
    """

    def __init__(self, model_path: str = LOCAL_MODEL_PATH):
        self._mode = "heuristic"
        self._yolo = None

        # Try loading local YOLO model first
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(model_path)
                self._mode = "yolo_local"
                print(f"[SnakeDetector] ✅ Using local snake model: {model_path}")
                logger.info("[SnakeDetector] Using local model: %s", model_path)
                return
            except Exception as e:
                logger.warning("[SnakeDetector] Local model load failed: %s — using heuristic", e)

        # Fallback: heuristic
        print(
            "[SnakeDetector] ⚠️  No models/snake.pt found — using shape-heuristic fallback.\n"
            "   → For better accuracy: run  python scripts/download_snake_model.py"
        )
        logger.warning("[SnakeDetector] Running in heuristic mode (no snake.pt found)")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(self, frame) -> List[ThreatAssessment]:
        """Run detection on a single frame and return ThreatAssessments."""
        if frame is None:
            return []
        if self._mode == "yolo_local":
            return self._detect_yolo(frame)
        return self._detect_heuristic(frame)

    # -----------------------------------------------------------------------
    # Internal strategies
    # -----------------------------------------------------------------------

    def _detect_yolo(self, frame) -> List[ThreatAssessment]:
        results     = self._yolo(frame, conf=CONF_THRESHOLD, verbose=False)
        assessments = []
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                obj  = result.names[cls]
                conf = float(box.conf[0])
                if "snake" in obj.lower():
                    assessments.append(assess(obj="snake", confidence=conf))
        return assessments

    def _detect_heuristic(self, frame) -> List[ThreatAssessment]:
        raw         = _heuristic_snake_detect(frame)
        assessments = []
        now         = datetime.now(timezone.utc).isoformat()
        for det in raw:
            if det["confidence"] >= CONF_THRESHOLD:
                assessments.append(
                    ThreatAssessment(
                        severity    = Severity.HIGH,
                        category    = EventCategory.SNAKE,
                        label       = "[HIGH] Snake Detected",
                        confidence  = det["confidence"],
                        description = "Elongated snake-like shape detected. Verify on camera feed.",
                        timestamp   = now,
                    )
                )
        return assessments
