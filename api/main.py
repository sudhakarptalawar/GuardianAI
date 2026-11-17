"""Local GuardianAI dashboard API and camera runtime."""
from __future__ import annotations

import threading
import time
import csv
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.event_manager import EventManager
from core.clip_manager import ClipManager
from core.alert_manager import AlertManager
from core.auth import (
    ensure_users_table, create_user, authenticate_user,
    get_user_by_id, update_user, change_password,
    create_token, decode_token
)
from detectors.fire import FireDetector
from detectors.objects import ObjectDetector
from detectors.person import PersonDetector
from detectors.snake import SnakeDetector
from detectors.face import FaceDetector
from detectors.perimeter import PerimeterDetector

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
EVENTS = EventManager(str(ROOT / "reports" / "events.json"))
AVAILABLE = {"person", "fire", "object", "snake", "face", "perimeter"}
DEMO_CLIPS = {
    "general": {"label": "People, animals & vehicles", "path": "videos/test.mp4", "detectors": ["person", "object"]},
    "fire": {"label": "Fire & smoke", "path": "videos/fire/fire.mp4", "detectors": ["person", "fire"]},
    "snake": {"label": "Snake hazard", "path": "videos/snake/snake.mp4", "detectors": ["snake"]},
}


class StartRequest(BaseModel):
    source: str = "0"
    detectors: list[str] = Field(default_factory=lambda: ["person", "fire", "object"])
    camera_name: str = "Camera 0"
    location: str = "Front entry"
    fps: int = Field(default=5, ge=1, le=15)
    loop_video: bool = False


class DemoRequest(BaseModel):
    demo_id: str


class CameraRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._detectors: list[Any] = []
        self._detector_names: list[str] = []
        self._source: str | int = 0
        self._camera_name = "Camera 0"
        self._location = "Front entry"
        self._target_fps = 5
        self._loop_video = False
        self._fps = 0.0
        self._connected = False
        self._message = "Camera is stopped"
        self._last_error = ""
        self._labels: list[str] = []
        self._latest_severity = ""
        self._last_event_at = 0.0
        self._clip_manager = ClipManager()
        self._alert_manager = AlertManager()

    def start(self, request: StartRequest) -> None:
        invalid = set(request.detectors) - AVAILABLE
        if invalid:
            raise HTTPException(400, f"Unsupported detectors: {', '.join(sorted(invalid))}")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise HTTPException(409, "The camera is already running")
            source_text = request.source.strip() or "0"
            if source_text.startswith("videos/") or source_text.startswith("videos\\"):
                self._source = str((ROOT / source_text).resolve())
            else:
                self._source = int(source_text) if source_text.isdigit() else source_text
            self._loop_video = request.loop_video
            self._camera_name = request.camera_name.strip() or "Camera 0"
            self._location = request.location.strip() or "Unknown"
            self._target_fps = request.fps
            self._detector_names = request.detectors
            self._jpeg = None
            self._labels = []
            self._latest_severity = ""
            self._last_error = ""
            self._connected = False
            self._message = "Starting camera and loading detectors…"
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="guardianai-camera", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if not thread or not thread.is_alive():
                self._connected = False
                self._message = "Camera is stopped"
                self._jpeg = None
                return
            self._message = "Stopping camera…"
            self._stop.set()
        thread.join(timeout=4)
        with self._lock:
            if not thread.is_alive():
                self._connected = False
                self._message = "Camera is stopped"
                self._jpeg = None

    def _build_detectors(self) -> list[Any]:
        models = ROOT / "models"
        general_model = ROOT / "yolov8n.pt"
        detectors: list[Any] = []
        for name in self._detector_names:
            if name == "person":
                detectors.append(PersonDetector(model_path=str(general_model)))
            elif name == "fire":
                fire_model = models / "fire.pt"
                detectors.append(FireDetector(model_path=str(fire_model if fire_model.exists() else general_model)))
            elif name == "object":
                detectors.append(ObjectDetector(model_path=str(general_model)))
            elif name == "snake":
                detectors.append(SnakeDetector(model_path=str(general_model)))
            elif name == "face":
                detectors.append(FaceDetector())
            elif name == "perimeter":
                detectors.append(PerimeterDetector(model_path=str(general_model)))
        return detectors

    def _run(self) -> None:
        cap = None
        try:
            self._detectors = self._build_detectors()
            backend = cv2.CAP_DSHOW if isinstance(self._source, int) and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self._source, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera source: {self._source}")
            with self._lock:
                self._connected = True
                self._message = "Live feed connected"

            interval = 1.0 / max(self._target_fps, 1)
            last_inference = 0.0
            last_tick = time.monotonic()
            fps_samples = 0
            active_labels: list[str] = []
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self._loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera frame read failed")
                self._clip_manager.push(frame, fps=self._fps)
                now = time.monotonic()
                fps_samples += 1
                if now - last_tick >= 1.0:
                    with self._lock:
                        self._fps = round(fps_samples / max(now - last_tick, 0.001), 1)
                    fps_samples = 0
                    last_tick = now

                new_events = []
                if now - last_inference >= interval:
                    active_labels = []
                    for detector in self._detectors:
                        try:
                            assessments = detector.detect(frame)
                            for assessment in assessments:
                                active_labels.append(f"{assessment.category.value.title()} {assessment.confidence:.0%}")
                                event = EVENTS.add(assessment, camera_id="cam_0")
                                if event:
                                    new_events.append(event)
                                    if event.severity in {"HIGH", "CRITICAL"}:
                                        clip_path = self._clip_manager.save(event.id, label=event.label)
                                        if clip_path:
                                            event.clip_path = clip_path
                                        self._alert_manager.send(event)
                        except Exception as exc:
                            with self._lock:
                                self._last_error = f"Detector issue: {exc}"
                    last_inference = now
                    with self._lock:
                        self._labels = active_labels[:6]
                        if new_events:
                            self._latest_severity = new_events[-1].severity
                            self._last_event_at = time.time()

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 46), (9, 19, 34), -1)
                cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
                cv2.putText(frame, f"{self._camera_name}  |  LIVE  |  {self._fps:.0f} FPS", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (224, 238, 247), 2, cv2.LINE_AA)
                if active_labels:
                    cv2.putText(frame, "  •  ".join(active_labels[:3]), (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 230, 197), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Monitoring selected detections", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (164, 185, 201), 1, cv2.LINE_AA)
                encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    with self._lock:
                        self._jpeg = buffer.tobytes()
                time.sleep(0.005)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._connected = False
                self._message = f"Camera stopped: {exc}"
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._connected = False
                if self._stop.is_set():
                    self._message = "Camera is stopped"
                self._thread = None
                self._jpeg = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            recent_alert = self._latest_severity in {"HIGH", "CRITICAL"} and time.time() - self._last_event_at < 20
            return {
                "running": running,
                "connected": self._connected,
                "message": self._message,
                "last_error": self._last_error,
                "camera_name": self._camera_name,
                "location": self._location,
                "source": str(self._source),
                "fps": self._fps,
                "detectors": self._detector_names,
                "detections": list(self._labels),
                "all_safe": not recent_alert,
            }

    def stream(self):
        last_frame = None
        while True:
            with self._lock:
                frame = self._jpeg
                running = bool(self._thread and self._thread.is_alive())
            if frame and frame != last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            elif not running:
                return
            else:
                time.sleep(0.04)


runtime = CameraRuntime()
app = FastAPI(title="GuardianAI Local Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure users table exists on startup
ensure_users_table()


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _safe_user(user: dict) -> dict:
    """Strip password before returning user data to client."""
    return {k: v for k, v in user.items() if k != "password"}

def _require_auth(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    payload = decode_token(authorization[7:])
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(401, "User not found")
    return user


# ── Auth request models ───────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name:       str
    email:      str
    password:   str
    org:        str = ""
    role:       str = "operator"
    avatar_col: str = "#4ce0c0"

class LoginRequest(BaseModel):
    email:    str
    password: str

class UpdateProfileRequest(BaseModel):
    name:       Optional[str] = None
    org:        Optional[str] = None
    avatar_col: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    user = create_user(
        name=req.name, email=req.email, password=req.password,
        role=req.role, org=req.org, avatar_col=req.avatar_col,
    )
    if not user:
        raise HTTPException(409, "Email already registered")
    token = create_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": _safe_user(user)}

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = create_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": _safe_user(user)}

@app.get("/api/auth/me")
def me(authorization: Optional[str] = Header(None)):
    user = _require_auth(authorization)
    return _safe_user(user)

@app.patch("/api/auth/profile")
def update_profile(req: UpdateProfileRequest, authorization: Optional[str] = Header(None)):
    user = _require_auth(authorization)
    updated = update_user(user["id"], req.dict(exclude_none=True))
    return _safe_user(updated)

@app.post("/api/auth/change-password")
def change_pw(req: ChangePasswordRequest, authorization: Optional[str] = Header(None)):
    user = _require_auth(authorization)
    if not __import__("core.auth", fromlist=["verify_password"]).verify_password(req.current_password, user["password"]):
        raise HTTPException(400, "Current password is incorrect")
    change_password(user["id"], req.new_password)
    return {"ok": True}


@app.get("/api/status")
def get_status():
    status = runtime.status()
    all_events = EVENTS.get_all()
    today = date.today().isoformat()
    today_events = [event for event in all_events if event.timestamp.startswith(today)]
    summary = EVENTS.summary()
    status["today_events"] = len(today_events)
    status["total_events"] = summary["total_events"]
    status["high_priority"] = sum(1 for event in today_events if event.severity in {"HIGH", "CRITICAL"})
    return status


@app.get("/api/events")
def get_events(limit: int = 12):
    limit = min(max(limit, 1), 50)
    events = EVENTS.get_all()
    return [event.__dict__ for event in reversed(events[-limit:])]


@app.get("/api/demos")
def get_demos():
    return [
        {"id": demo_id, "label": clip["label"]}
        for demo_id, clip in DEMO_CLIPS.items()
        if (ROOT / clip["path"]).is_file()
    ]


@app.post("/api/demo/start")
def start_demo(request: DemoRequest):
    clip = DEMO_CLIPS.get(request.demo_id)
    if not clip or not (ROOT / clip["path"]).is_file():
        raise HTTPException(404, "That demo video is not available")
    runtime.start(StartRequest(
        source=clip["path"],
        detectors=clip["detectors"],
        camera_name=f"Demo: {clip['label']}",
        location="Demo video",
        loop_video=True,
    ))
    return {"ok": True, "message": f"Playing demo: {clip['label']}"}


@app.get("/api/search")
def search_events(q: str = "", limit: int = 100):
    """Search saved event metadata with plain-language time and object terms."""
    query = " ".join(q.lower().split())
    limit = min(max(limit, 1), 200)
    events = list(reversed(EVENTS.get_all()))
    if not query:
        return {"query": q, "count": 0, "results": []}

    now = datetime.now()
    only_today = "today" in query or "this morning" in query or "this evening" in query
    only_yesterday = "yesterday" in query
    after_midnight = "after midnight" in query or "before dawn" in query
    part_of_day = None
    if "this morning" in query:
        part_of_day = (5, 12)
    elif "this evening" in query or "tonight" in query:
        part_of_day = (18, 24)

    aliases = {
        "people": ("person", "people", "human"),
        "person": ("person", "people", "human"),
        "cars": ("car", "vehicle", "truck", "bus", "motorcycle"),
        "vehicles": ("car", "vehicle", "truck", "bus", "motorcycle"),
        "fire": ("fire", "flame"),
        "smoke": ("smoke", "fire"),
        "dogs": ("dog", "animal"),
        "dog": ("dog", "animal"),
        "packages": ("package", "parcel"),
        "parcel": ("package", "parcel"),
    }
    ignored = {"show", "find", "all", "events", "event", "detected", "detection", "camera", "cameras", "the", "a", "an", "in", "on", "at", "from", "me", "my", "of", "after", "before", "midnight", "today", "yesterday", "this", "morning", "evening", "tonight"}
    terms = [word.strip(".,!?\"'") for word in query.split()]
    terms = [word for word in terms if word and word not in ignored]
    selected = []
    for event in events:
        try:
            stamp = datetime.strptime(event.timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            stamp = now
        if only_today and stamp.date() != now.date():
            continue
        if only_yesterday and stamp.date() != (now.date() - timedelta(days=1)):
            continue
        if after_midnight and stamp.hour >= 6:
            continue
        if part_of_day and not (part_of_day[0] <= stamp.hour < part_of_day[1]):
            continue

        searchable = " ".join((event.label, event.category, event.description, event.camera_id, event.severity)).lower()
        matches = True
        for term in terms:
            candidates = aliases.get(term, (term,))
            if not any(candidate in searchable for candidate in candidates):
                matches = False
                break
        if matches:
            selected.append(event.__dict__)
            if len(selected) >= limit:
                break
    return {"query": q, "count": len(selected), "results": selected}


@app.get("/api/reports/summary")
def report_summary(days: int = 7):
    days = min(max(days, 1), 90)
    events = EVENTS.get_all()
    today = date.today()
    daily = []
    for offset in reversed(range(days)):
        day = today - timedelta(days=offset)
        daily.append({"date": day.isoformat(), "count": sum(1 for event in events if event.timestamp.startswith(day.isoformat()))})
    severity = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    categories: dict[str, int] = {}
    for event in events:
        severity[event.severity] = severity.get(event.severity, 0) + 1
        categories[event.category] = categories.get(event.category, 0) + 1
    return {
        "total_events": len(events),
        "today_events": sum(1 for event in events if event.timestamp.startswith(today.isoformat())),
        "severity": severity,
        "categories": categories,
        "daily": daily,
    }


@app.get("/api/reports/export")
def export_report():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "camera", "category", "severity", "event", "confidence", "description"])
    for event in reversed(EVENTS.get_all()):
        writer.writerow([event.timestamp, event.camera_id, event.category, event.severity, event.label, event.confidence, event.description])
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=guardianai-events.csv"})


@app.post("/api/camera/start")
def start_camera(request: StartRequest):
    runtime.start(request)
    return {"ok": True, "message": "Camera startup requested"}


@app.post("/api/camera/stop")
def stop_camera():
    runtime.stop()
    return {"ok": True, "message": "Camera stopped"}


@app.get("/api/stream")
def camera_stream():
    return StreamingResponse(runtime.stream(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})


@app.on_event("shutdown")
def stop_on_shutdown():
    runtime.stop()


@app.get("/api/clips/{clip_name}")
def get_clip(clip_name: str):
    clip_path = (ROOT / "reports" / "clips" / clip_name).resolve()
    if not clip_path.is_file():
        raise HTTPException(404, "Evidence clip not found")
    return FileResponse(clip_path, media_type="video/mp4")


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
