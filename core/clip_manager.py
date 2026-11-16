"""
GuardianAI — Evidence Clip Manager
Auto-saves a short video clip around each detection event.
Keeps a rolling buffer of recent frames and writes ±N seconds on demand.
"""

import cv2
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional


CLIPS_DIR    = "reports/clips"
BUFFER_SEC   = 5       # seconds of frames to keep before the event
POST_SEC     = 3       # seconds to record after the event


class ClipManager:
    """
    Maintains a rolling frame buffer and saves evidence clips on request.

    Usage:
        clips = ClipManager()
        clips.push(frame, fps=5.0)       # call every frame
        path = clips.save("EVT-001")     # call when an event fires
    """

    def __init__(
        self,
        clips_dir:  str   = CLIPS_DIR,
        buffer_sec: int   = BUFFER_SEC,
        post_sec:   int   = POST_SEC,
        fps:        float = 5.0,
    ):
        self.clips_dir  = clips_dir
        self.buffer_sec = buffer_sec
        self.post_sec   = post_sec
        self.fps        = fps
        self._buffer: deque = deque()
        self._lock    = threading.Lock()
        self._active_clips: dict[str, dict] = {}  # event_id → recording state
        os.makedirs(clips_dir, exist_ok=True)

    def push(self, frame, fps: float = 5.0):
        """Add a frame to the rolling buffer and feed active recordings."""
        self.fps     = fps or self.fps
        max_frames   = int(self.buffer_sec * max(self.fps, 1))
        ts           = time.time()

        with self._lock:
            self._buffer.append((ts, frame.copy()))
            # Trim old frames
            while len(self._buffer) > max_frames:
                self._buffer.popleft()

            # Feed any active post-event recordings
            for rec in list(self._active_clips.values()):
                if rec["writer"] and not rec["done"]:
                    rec["writer"].write(frame)
                    rec["frames_written"] += 1
                    if rec["frames_written"] >= rec["post_frames"]:
                        rec["writer"].release()
                        rec["done"] = True

    def save(self, event_id: str, label: str = "") -> Optional[str]:
        """
        Save a clip for an event.
        Pre-event frames come from the buffer; post-event frames are streamed in via push().
        Returns the output file path (or None on failure).
        """
        ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id   = event_id.replace("/", "_").replace(":", "_")
        out_path  = os.path.join(self.clips_dir, f"clip_{safe_id}_{ts_str}.mp4")

        with self._lock:
            if not self._buffer:
                return None

            # Collect pre-event frames from buffer
            pre_frames = [(ts, f.copy()) for ts, f in self._buffer]
            if not pre_frames:
                return None

            h, w = pre_frames[0][1].shape[:2]
            fps_int = max(int(self.fps), 1)

            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_int,
                (w, h),
            )

            for _, f in pre_frames:
                writer.write(f)

            post_frames = int(self.post_sec * fps_int)

            self._active_clips[event_id] = {
                "writer":        writer,
                "post_frames":   post_frames,
                "frames_written": 0,
                "done":          False,
                "path":          out_path,
            }

        print(f"[ClipManager] 🎬 Saving clip → {out_path}")
        return out_path

    def get_clip_path(self, event_id: str) -> Optional[str]:
        """Return the saved clip path for an event (after recording finishes)."""
        rec = self._active_clips.get(event_id)
        return rec["path"] if rec and rec["done"] else None

    def cleanup(self, max_clips: int = 50):
        """Remove oldest clips if more than max_clips exist."""
        clips = sorted(
            [f for f in os.listdir(self.clips_dir) if f.endswith(".mp4")],
            key=lambda f: os.path.getmtime(os.path.join(self.clips_dir, f)),
        )
        for old in clips[:-max_clips]:
            try:
                os.remove(os.path.join(self.clips_dir, old))
            except Exception:
                pass
