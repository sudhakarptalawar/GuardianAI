"""
GuardianAI — Unified Detection Pipeline
Runs all enabled detectors on each camera frame and routes results
to the EventManager and AlertManager.
"""

import cv2
import time
from typing import List, Callable

from core.camera       import Camera, CameraConfig
from core.severity     import ThreatAssessment
from core.event_manager import EventManager, Event
from core.clip_manager import ClipManager
from core.alert_manager import AlertManager


# Type alias: a detector is any callable that takes a frame and returns a list of assessments
DetectorFn = Callable[[object], List[ThreatAssessment]]


class Pipeline:
    """
    GuardianAI unified detection pipeline.

    Usage:
        pipeline = Pipeline(camera_id="cam_0", source=0)
        pipeline.add_detector(person_detector.detect)
        pipeline.add_detector(fire_detector.detect)
        pipeline.run()          # blocking
        pipeline.run(show=True) # with live preview window
    """

    def __init__(
        self,
        camera_id: str         = "cam_0",
        source:    str | int   = 0,
        name:      str         = "Camera 0",
        location:  str         = "Unknown",
        fps_target: int        = 5,
        event_manager: EventManager | None = None,
        on_event: Callable[[Event], None] | None = None,
    ):
        self.camera_id      = camera_id
        self.source         = source
        self.name           = name
        self.location       = location
        self.fps_target     = fps_target
        self.detectors: List[DetectorFn] = []
        self.event_manager  = event_manager or EventManager()
        self.clip_manager   = ClipManager()
        self.alert_manager  = AlertManager()
        self.on_event       = on_event      # optional callback (e.g. send alert)
        self._running       = False

    # ── Detector registry ─────────────────────────────────────────────────────

    def add_detector(self, detector_fn: DetectorFn) -> "Pipeline":
        """Register a detector function. Returns self for chaining."""
        self.detectors.append(detector_fn)
        return self

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, show: bool = False, duration: float | None = None):
        """
        Start the pipeline. Blocks until the stream ends or duration (seconds) elapses.

        Args:
            show:     If True, opens an OpenCV preview window with detections overlay.
            duration: Optional max run time in seconds (useful for testing).
        """
        config = CameraConfig(
            camera_id  = self.camera_id,
            source     = self.source,
            name       = self.name,
            location   = self.location,
            fps_target = self.fps_target,
        )
        cam = Camera(config)

        if not cam.start():
            print(f"[Pipeline] ❌ Could not start camera '{self.camera_id}'. Aborting.")
            return

        self._running = True
        start_time    = time.time()
        print(f"\n[Pipeline] 🚀 Running on camera '{self.name}' with {len(self.detectors)} detector(s).")
        print("[Pipeline] Press Ctrl+C to stop.\n")

        try:
            for frame in cam.frames():
                if not self._running:
                    break
                if duration and (time.time() - start_time) > duration:
                    break

                self.clip_manager.push(frame, fps=cam.status.fps)
                annotated = frame.copy()
                all_events: List[Event] = []

                for detector in self.detectors:
                    try:
                        assessments: List[ThreatAssessment] = detector(frame)
                        for assessment in assessments:
                            event = self.event_manager.add(
                                assessment = assessment,
                                camera_id  = self.camera_id,
                            )
                            if event:
                                all_events.append(event)
                                if event.severity in ("HIGH", "CRITICAL"):
                                    clip_path = self.clip_manager.save(event.id, label=event.label)
                                    if clip_path:
                                        event.clip_path = clip_path
                                    self.alert_manager.send(event)
                                if self.on_event:
                                    self.on_event(event)
                    except Exception as ex:
                        print(f"[Pipeline] ⚠️  Detector error: {ex}")

                if show:
                    annotated = self._annotate(annotated, all_events, cam.status)
                    cv2.imshow(f"GuardianAI — {self.name}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[Pipeline] Stopped by user.")
        finally:
            self._running = False
            cam.stop()
            if show:
                cv2.destroyAllWindows()
            self._print_summary()

    def stop(self):
        """Signal the pipeline to stop after the current frame."""
        self._running = False

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame, events: List[Event], status) -> object:
        """Overlay event labels and camera status on the frame."""
        COLORS = {
            "CRITICAL": (0,   0,   255),
            "HIGH":     (0,   100, 255),
            "MEDIUM":   (0,   200, 255),
            "LOW":      (0,   255, 0),
        }
        # Camera name + FPS
        cv2.putText(
            frame,
            f"{status.name}  |  {status.fps} FPS",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )

        # Active events overlay
        y = 55
        for event in events[-5:]:   # show last 5 events max
            color = COLORS.get(event.severity, (200, 200, 200))
            cv2.putText(
                frame,
                event.label,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            y += 25

        return frame

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        summary = self.event_manager.summary()
        print("\n" + "=" * 40)
        print("  GUARDIANAI SESSION SUMMARY")
        print("=" * 40)
        print(f"  Total Events : {summary['total_events']}")
        for severity, count in summary["by_severity"].items():
            if count:
                print(f"  {severity:<10}: {count}")
        print("=" * 40 + "\n")
