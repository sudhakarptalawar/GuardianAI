"""
GuardianAI — Event Manager
Handles creation, deduplication, and storage of detection events.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

from core.severity import ThreatAssessment, Severity


EVENTS_FILE = "reports/events.json"


@dataclass
class Event:
    id:          str
    timestamp:   str
    camera_id:   str
    category:    str
    severity:    str
    label:       str
    description: str
    confidence:  float
    clip_path:   str | None = None


class EventManager:
    """
    Manages the lifecycle of GuardianAI detection events.
    Stores events in memory and persists to a JSON file.
    """

    def __init__(self, events_file: str = EVENTS_FILE):
        self.events_file = events_file
        self.events: List[Event] = []
        self._recent_labels: dict[str, str] = {}   # camera_id → last label (dedup)
        os.makedirs(os.path.dirname(events_file), exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        assessment: ThreatAssessment,
        camera_id: str = "cam_0",
        clip_path: str | None = None,
    ) -> Event | None:
        """
        Create and store an event from a ThreatAssessment.
        Returns None if the same event was already logged within the last second
        (simple dedup to avoid flooding on video frames).
        """
        dedup_key = f"{camera_id}:{assessment.label}"
        last_ts   = self._recent_labels.get(dedup_key)

        # Deduplicate: skip if the exact same label was added < 3 s ago
        if last_ts:
            delta = (
                datetime.strptime(assessment.timestamp, "%Y-%m-%d %H:%M:%S")
                - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            ).total_seconds()
            if delta < 3:
                return None

        event = Event(
            id          = self._make_id(),
            timestamp   = assessment.timestamp,
            camera_id   = camera_id,
            category    = assessment.category.value,
            severity    = assessment.severity.value,
            label       = assessment.label,
            description = assessment.description,
            confidence  = round(assessment.confidence, 3),
            clip_path   = clip_path,
        )

        self.events.append(event)
        self._recent_labels[dedup_key] = assessment.timestamp
        self._save()
        self._print(event)
        return event

    def get_all(self) -> List[Event]:
        return list(self.events)

    def get_by_severity(self, severity: Severity) -> List[Event]:
        return [e for e in self.events if e.severity == severity.value]

    def get_by_camera(self, camera_id: str) -> List[Event]:
        return [e for e in self.events if e.camera_id == camera_id]

    def summary(self) -> dict:
        counts = {s.value: 0 for s in Severity}
        for e in self.events:
            counts[e.severity] = counts.get(e.severity, 0) + 1
        return {
            "total_events": len(self.events),
            "by_severity":  counts,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_id(self) -> str:
        return f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _print(self, event: Event) -> None:
        icons = {
            "CRITICAL": "🔴",
            "HIGH":     "🟠",
            "MEDIUM":   "🟡",
            "LOW":      "🟢",
        }
        icon = icons.get(event.severity, "⚪")
        print(f"{icon} [{event.timestamp}] {event.label} | cam={event.camera_id} | conf={event.confidence}")

    def _save(self) -> None:
        try:
            with open(self.events_file, "w") as f:
                json.dump([asdict(e) for e in self.events], f, indent=2)
        except Exception as ex:
            print(f"[EventManager] Warning: could not save events — {ex}")

    def _load(self) -> None:
        if not os.path.exists(self.events_file):
            return
        try:
            with open(self.events_file) as f:
                data = json.load(f)
            self.events = [Event(**d) for d in data]
        except Exception as ex:
            print(f"[EventManager] Warning: could not load events — {ex}")
