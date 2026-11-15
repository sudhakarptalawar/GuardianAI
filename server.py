"""
GuardianAI — Unified API Server (matches frontend contract exactly)
Serves:
  - All /api/* REST endpoints the dashboard calls
  - MJPEG live stream at /api/stream
  - Static frontend files at /
"""

import os
import cv2
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from core.event_manager import EventManager
from core.severity       import Severity


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="GuardianAI API", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────

event_manager = EventManager()

_state = {
    "running":     False,
    "connected":   False,
    "fps":         0.0,
    "camera_name": "Camera 0",
    "location":    "Front entry",
    "detections":  [],        # list of active detection label strings
    "all_safe":    True,
    "last_error":  "",
    "message":     "",
    "today_events": 0,
    "high_priority": 0,
}

_latest_frame = None
_frame_lock   = threading.Lock()
_pipeline_thread: Optional[threading.Thread] = None
_stop_event   = threading.Event()


# ── Camera / pipeline control ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    source:      str  = "0"
    detectors:   list = ["person", "fire", "object"]
    camera_name: str  = "Camera 0"
    location:    str  = "Front entry"


def _run_pipeline(source, detectors, camera_name, location):
    """Background thread: runs the detection pipeline and updates shared state."""
    global _latest_frame

    from core.camera        import Camera, CameraConfig
    from core.severity      import assess
    from detectors.person   import PersonDetector
    from detectors.fire     import FireDetector
    from detectors.snake    import SnakeDetector
    from detectors.objects  import ObjectDetector

    DETECTOR_MAP = {
        "person": PersonDetector,
        "fire":   FireDetector,
        "snake":  SnakeDetector,
        "object": ObjectDetector,
    }

    _state["message"]     = "Loading detection models…"
    _state["last_error"]  = ""

    # Load detectors
    active_detectors = []
    for name in detectors:
        if name in DETECTOR_MAP:
            try:
                active_detectors.append(DETECTOR_MAP[name]())
            except Exception as ex:
                print(f"[API] Could not load detector '{name}': {ex}")

    # Open camera
    src = int(source) if str(source).isdigit() else source
    config = CameraConfig(source=src, name=camera_name, location=location, fps_target=5)
    cam    = Camera(config)

    if not cam.start():
        _state["running"]    = False
        _state["connected"]  = False
        _state["last_error"] = f"Could not open camera source: {source}"
        _state["message"]    = _state["last_error"]
        return

    _state["connected"]  = True
    _state["camera_name"] = camera_name
    _state["location"]   = location
    _state["message"]    = "Camera connected."

    try:
        for frame in cam.frames():
            if _stop_event.is_set():
                break

            # Store latest frame for MJPEG stream
            with _frame_lock:
                _latest_frame = frame.copy()

            _state["fps"] = cam.status.fps

            # Run detectors
            frame_detections = []
            person_count = 0

            for detector in active_detectors:
                try:
                    assessments = detector.detect(frame)
                    for a in assessments:
                        if a.label.lower().__contains__("person"):
                            person_count += 1
                        ev = event_manager.add(a, camera_id="cam_0")
                        if ev:
                            frame_detections.append(ev.label.replace("[CRITICAL] ", "")
                                                              .replace("[HIGH] ", "")
                                                              .replace("[MEDIUM] ", "")
                                                              .replace("[LOW] ", ""))
                except Exception as ex:
                    print(f"[API] Detector error: {ex}")

            # Update live state
            has_critical = any(
                e.severity in ("HIGH", "CRITICAL")
                for e in event_manager.get_all()
                if e.timestamp >= (datetime.now() - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")
            )
            _state["detections"]   = list(set(frame_detections))[:6]
            _state["all_safe"]     = not has_critical
            _state["today_events"] = len(event_manager.get_all())
            _state["high_priority"] = len(event_manager.get_by_severity(Severity.HIGH)) + \
                                       len(event_manager.get_by_severity(Severity.CRITICAL))

    finally:
        cam.stop()
        _state["running"]   = False
        _state["connected"] = False
        _state["fps"]       = 0.0
        _state["detections"] = []
        with _frame_lock:
            _latest_frame = None


@app.post("/api/camera/start")
def camera_start(req: StartRequest):
    global _pipeline_thread
    if _state["running"]:
        raise HTTPException(status_code=409, detail="Camera is already running.")

    _stop_event.clear()
    _state["running"] = True

    _pipeline_thread = threading.Thread(
        target=_run_pipeline,
        args=(req.source, req.detectors, req.camera_name, req.location),
        daemon=True,
    )
    _pipeline_thread.start()
    return {"success": True, "message": f"Starting '{req.camera_name}' on source {req.source}…"}


@app.post("/api/camera/stop")
def camera_stop():
    _stop_event.set()
    _state["running"]   = False
    _state["connected"] = False
    _state["message"]   = "Camera stopped."
    return {"success": True, "message": "Camera stopped."}


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return {**_state}


# ── Events ────────────────────────────────────────────────────────────────────

@app.get("/api/events")
def api_events(
    limit:     int           = Query(50, ge=1, le=1000),
    severity:  Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
):
    events = event_manager.get_all()

    if severity:
        events = [e for e in events if e.severity == severity.upper()]
    if camera_id:
        events = [e for e in events if e.camera_id == camera_id]

    events = list(reversed(events))[:limit]
    return [e.__dict__ for e in events]


@app.delete("/api/events")
def api_clear_events():
    event_manager.events.clear()
    event_manager._save()
    return {"success": True, "message": "All events cleared."}


# ── Search ────────────────────────────────────────────────────────────────────

@app.get("/api/search")
def api_search(q: str = Query(..., description="Search query")):
    query  = q.strip().lower()
    events = event_manager.get_all()

    results = [
        e for e in events
        if any(
            query in str(getattr(e, field, "")).lower()
            for field in ("label", "category", "description", "camera_id", "severity", "timestamp")
        )
    ]

    results = list(reversed(results))[:100]
    return {
        "query":   q,
        "count":   len(results),
        "results": [e.__dict__ for e in results],
    }


# ── Reports ───────────────────────────────────────────────────────────────────

@app.get("/api/reports/summary")
def api_reports_summary(days: int = Query(7, ge=1, le=90)):
    events = event_manager.get_all()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    events = [e for e in events if e.timestamp >= cutoff]

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_events = len([e for e in events if e.timestamp.startswith(today_str)])

    # Severity counts
    severity_counts = {s.value: 0 for s in Severity}
    for e in events:
        severity_counts[e.severity] = severity_counts.get(e.severity, 0) + 1

    # Category counts
    category_counts: dict = {}
    for e in events:
        category_counts[e.category] = category_counts.get(e.category, 0) + 1

    # Daily counts (last N days)
    daily = []
    for i in range(days - 1, -1, -1):
        day     = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        count   = len([e for e in events if e.timestamp.startswith(day)])
        daily.append({"date": day, "count": count})

    return {
        "total_events":  len(events),
        "today_events":  today_events,
        "severity":      severity_counts,
        "categories":    category_counts,
        "daily":         daily,
    }


# ── Demo videos ───────────────────────────────────────────────────────────────

def _scan_demos() -> list:
    """Scan videos/ folder and return demo entries."""
    demos = []
    videos_dir = "videos"
    if not os.path.isdir(videos_dir):
        return demos
    for root, _, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                rel  = os.path.join(root, f).replace("\\", "/")
                name = os.path.splitext(f)[0].replace("_", " ").replace("-", " ").title()
                demos.append({"id": rel, "label": name, "path": rel})
    return demos


@app.get("/api/demos")
def api_demos():
    return _scan_demos()


class DemoStartRequest(BaseModel):
    demo_id: str


@app.post("/api/demo/start")
def api_demo_start(req: DemoStartRequest):
    global _pipeline_thread
    if _state["running"]:
        raise HTTPException(status_code=409, detail="Stop the current camera first.")

    # Validate path
    demos = {d["id"]: d for d in _scan_demos()}
    if req.demo_id not in demos:
        raise HTTPException(status_code=404, detail="Demo video not found.")

    video_path = demos[req.demo_id]["path"]
    label      = demos[req.demo_id]["label"]

    _stop_event.clear()
    _state["running"]     = True
    _state["camera_name"] = label
    _state["location"]    = "Demo"

    _pipeline_thread = threading.Thread(
        target=_run_pipeline,
        args=(video_path, ["person", "fire", "snake", "object"], label, "Demo"),
        daemon=True,
    )
    _pipeline_thread.start()
    return {"success": True, "message": f"Playing demo: {label}"}


# ── MJPEG live stream ─────────────────────────────────────────────────────────

def _mjpeg_generator():
    """Yield MJPEG frames from the latest captured frame."""
    boundary = b"--frame"
    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() + b"\r\n"
            )
        else:
            time.sleep(0.1)
        time.sleep(0.04)   # ~25 FPS cap for stream


@app.get("/api/stream")
def api_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Alerts (stub — full delivery in Week 3) ───────────────────────────────────

@app.get("/api/alerts/config")
def alerts_config():
    config_path = "reports/alert_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {"enabled": True, "min_severity": "HIGH", "email": None, "phone": None}


@app.post("/api/alerts/config")
def alerts_config_save(config: dict):
    os.makedirs("reports", exist_ok=True)
    with open("reports/alert_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return {"success": True}


@app.post("/api/alerts/test")
def alerts_test():
    return {"success": True, "message": "Test alert sent (email delivery coming in Week 3)."}


# ── Serve frontend static files ───────────────────────────────────────────────

if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
