"""
GuardianAI — Camera Module
Handles live webcam and RTSP stream ingestion with frame sampling.
"""

import cv2
import threading
import time
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class CameraConfig:
    camera_id:    str   = "cam_0"
    source:       str | int = 0          # 0 = default webcam, or RTSP URL string
    fps_target:   int   = 5             # frames per second to process (not capture)
    width:        int   = 640
    height:       int   = 480
    name:         str   = "Camera 0"
    location:     str   = "Unknown"


@dataclass
class CameraStatus:
    camera_id:  str
    name:       str
    connected:  bool  = False
    fps:        float = 0.0
    frame_count: int  = 0
    last_error: str   = ""


class Camera:
    """
    Thread-safe camera reader that captures frames from a webcam or RTSP stream.
    Use as a context manager or call start()/stop() manually.

    Example:
        cam = Camera(CameraConfig(source=0))
        with cam:
            for frame in cam.frames():
                process(frame)
    """

    def __init__(self, config: CameraConfig):
        self.config  = config
        self._cap    = None
        self._lock   = threading.Lock()
        self._latest = None
        self._running = False
        self._thread  = None
        self.status   = CameraStatus(
            camera_id = config.camera_id,
            name      = config.name,
        )

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the stream and start the background capture thread."""
        self._cap = cv2.VideoCapture(self.config.source)
        if not self._cap.isOpened():
            self.status.connected = False
            self.status.last_error = f"Cannot open source: {self.config.source}"
            print(f"[Camera {self.config.camera_id}] ❌ {self.status.last_error}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.status.connected = True
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera {self.config.camera_id}] ✅ Connected — {self.config.name} @ {self.config.source}")
        return True

    def stop(self):
        """Stop capture thread and release the stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        self.status.connected = False
        print(f"[Camera {self.config.camera_id}] 🔌 Disconnected.")

    # ── Frame access ─────────────────────────────────────────────────────────

    def read(self):
        """Return the latest captured frame (or None if not yet available)."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def frames(self, timeout: float = 30.0) -> Generator:
        """
        Generator that yields frames at the configured fps_target rate.
        Stops after `timeout` seconds of no new frame (stream stall).
        """
        interval     = 1.0 / max(self.config.fps_target, 1)
        last_seen    = time.time()
        while self._running:
            frame = self.read()
            if frame is not None:
                last_seen = time.time()
                yield frame
            elif time.time() - last_seen > timeout:
                print(f"[Camera {self.config.camera_id}] ⚠️  Stream stalled. Stopping.")
                break
            time.sleep(interval)

    # ── Background capture loop ───────────────────────────────────────────────

    def _capture_loop(self):
        fps_counter = 0
        fps_timer   = time.time()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.status.connected = False
                self.status.last_error = "Stream read failed."
                print(f"[Camera {self.config.camera_id}] ⚠️  Read failed — retrying in 1s…")
                time.sleep(1)
                # Attempt reconnect
                self._cap.release()
                self._cap = cv2.VideoCapture(self.config.source)
                continue

            with self._lock:
                self._latest = frame

            self.status.frame_count += 1
            fps_counter += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.status.fps = round(fps_counter / elapsed, 1)
                fps_counter      = 0
                fps_timer        = time.time()


# ── Convenience helper ────────────────────────────────────────────────────────

def open_camera(source: str | int = 0, camera_id: str = "cam_0", name: str = "Camera") -> Camera:
    """Quick helper to create and start a camera."""
    cfg = CameraConfig(source=source, camera_id=camera_id, name=name)
    cam = Camera(cfg)
    cam.start()
    return cam
