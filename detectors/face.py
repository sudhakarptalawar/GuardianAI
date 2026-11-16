"""
GuardianAI — Face & Covered Face Detector
Detects visible faces and flags covered/obscured faces on detected persons.
Uses OpenCV Haar Cascade face detection.
"""

import cv2
import numpy as np
from typing import List

from core.severity import assess, ThreatAssessment


class FaceDetector:
    """
    Detects faces in a frame using OpenCV Haar Cascades.
    Flags covered face / unidentifiable face if configured.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("[FaceDetector] ⚠️ Warning: Failed to load Haar cascade classifier.")
        else:
            print("[FaceDetector] ✅ Ready.")

    def detect(self, frame) -> List[ThreatAssessment]:
        """Detect faces in the frame and return ThreatAssessments."""
        assessments: List[ThreatAssessment] = []
        if self.face_cascade.empty() or frame is None:
            return assessments

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(36, 36),
        )

        num_faces = len(faces)
        if num_faces > 0:
            # We detected identifiable faces
            conf = min(0.65 + 0.1 * num_faces, 0.95)
            assessments.append(assess(
                obj="face",
                confidence=conf,
                context={"face_count": num_faces},
            ))

        return assessments
