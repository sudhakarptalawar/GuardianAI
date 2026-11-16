"""
GuardianAI — Severity Scoring Engine
Maps detection objects + context to severity levels and event categories.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class EventCategory(str, Enum):
    PERSON        = "PERSON"
    INTRUDER      = "INTRUDER"
    FIRE          = "FIRE"
    SMOKE         = "SMOKE"
    SNAKE         = "SNAKE"
    ANIMAL        = "ANIMAL"
    VEHICLE       = "VEHICLE"
    FALL          = "FALL"
    FIGHT         = "FIGHT"
    COVERED_FACE  = "COVERED_FACE"
    UNKNOWN       = "UNKNOWN"


@dataclass
class ThreatAssessment:
    category:    EventCategory
    severity:    Severity
    label:       str
    confidence:  float
    description: str
    timestamp:   str


def is_night_time(hour: int | None = None) -> bool:
    """Returns True if current hour is between 10 PM and 5 AM."""
    h = hour if hour is not None else datetime.now().hour
    return h >= 22 or h < 5


def assess(obj: str, confidence: float = 1.0, context: dict | None = None) -> ThreatAssessment:
    """
    Given a detected object label and optional context, return a ThreatAssessment.

    Args:
        obj:        YOLO class label (e.g. "person", "fire", "snake")
        confidence: Detection confidence 0.0–1.0
        context:    Optional dict with extra context keys:
                      - person_count (int)   : how many persons already seen this frame
                      - hour (int)           : override current hour for testing
    """
    ctx       = context or {}
    obj_lower = obj.lower()
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hour      = ctx.get("hour", datetime.now().hour)
    night     = is_night_time(hour)

    # ── Person ──────────────────────────────────────────────────────────────
    if obj_lower == "person":
        person_count = ctx.get("person_count", 0)
        if night:
            return ThreatAssessment(
                category    = EventCategory.INTRUDER,
                severity    = Severity.HIGH,
                label       = "[HIGH] Potential Intruder Detected",
                confidence  = confidence,
                description = f"Person detected at night ({hour:02d}:xx). Possible intrusion.",
                timestamp   = now,
            )
        elif person_count > 2:
            return ThreatAssessment(
                category    = EventCategory.PERSON,
                severity    = Severity.MEDIUM,
                label       = "[MEDIUM] Multiple Persons Detected",
                confidence  = confidence,
                description = f"{person_count} persons detected in frame.",
                timestamp   = now,
            )
        else:
            return ThreatAssessment(
                category    = EventCategory.PERSON,
                severity    = Severity.LOW,
                label       = "[LOW] Person Detected",
                confidence  = confidence,
                description = "Person visible in camera view.",
                timestamp   = now,
            )

    # ── Fire & Smoke ─────────────────────────────────────────────────────────
    if obj_lower == "fire":
        return ThreatAssessment(
            category    = EventCategory.FIRE,
            severity    = Severity.CRITICAL,
            label       = "[CRITICAL] Fire Detected",
            confidence  = confidence,
            description = "Active fire detected in camera view. Immediate action required.",
            timestamp   = now,
        )

    if obj_lower == "smoke":
        return ThreatAssessment(
            category    = EventCategory.SMOKE,
            severity    = Severity.HIGH,
            label       = "[HIGH] Smoke Detected",
            confidence  = confidence,
            description = "Smoke detected. Possible fire risk.",
            timestamp   = now,
        )

    # ── Snake ────────────────────────────────────────────────────────────────
    if obj_lower == "snake":
        return ThreatAssessment(
            category    = EventCategory.SNAKE,
            severity    = Severity.HIGH,
            label       = "[HIGH] Snake Detected",
            confidence  = confidence,
            description = "Snake detected in camera view. Potential danger.",
            timestamp   = now,
        )

    # ── Face & Covered Face ──────────────────────────────────────────────────
    if obj_lower == "covered_face":
        return ThreatAssessment(
            category    = EventCategory.COVERED_FACE,
            severity    = Severity.HIGH,
            label       = "[HIGH] Concealed / Covered Face Detected",
            confidence  = confidence,
            description = "Individual with concealed face detected near camera.",
            timestamp   = now,
        )

    if obj_lower == "face":
        face_count = ctx.get("face_count", 1)
        return ThreatAssessment(
            category    = EventCategory.PERSON,
            severity    = Severity.LOW,
            label       = f"[LOW] Face Detected ({face_count})",
            confidence  = confidence,
            description = f"{face_count} face(s) identified in camera view.",
            timestamp   = now,
        )

    # ── Animals ──────────────────────────────────────────────────────────────
    if obj_lower in ("dog", "cat", "bear", "elephant"):
        return ThreatAssessment(
            category    = EventCategory.ANIMAL,
            severity    = Severity.LOW,
            label       = f"[LOW] {obj.capitalize()} Detected",
            confidence  = confidence,
            description = f"{obj.capitalize()} detected in camera view.",
            timestamp   = now,
        )

    # ── Vehicles ─────────────────────────────────────────────────────────────
    if obj_lower in ("car", "truck", "motorcycle", "bus"):
        severity = Severity.HIGH if night else Severity.MEDIUM
        label    = f"[{'HIGH' if night else 'MEDIUM'}] Vehicle Detected"
        return ThreatAssessment(
            category    = EventCategory.VEHICLE,
            severity    = severity,
            label       = label,
            confidence  = confidence,
            description = f"{obj.capitalize()} detected {'at night' if night else 'in view'}.",
            timestamp   = now,
        )

    # ── Default / Unknown ────────────────────────────────────────────────────
    return ThreatAssessment(
        category    = EventCategory.UNKNOWN,
        severity    = Severity.LOW,
        label       = f"[LOW] {obj.capitalize()} Detected",
        confidence  = confidence,
        description = f"Object detected: {obj}",
        timestamp   = now,
    )
