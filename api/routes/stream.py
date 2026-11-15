"""
GuardianAI — Stream API Route
WebSocket /stream/ws/{camera_id}  → real-time MJPEG frames as base64 JSON
GET       /stream/cameras         → list all registered cameras and their status
GET       /stream/status          → pipeline health check
"""

import cv2
import base64
import asyncio
import json
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from core.camera import CameraStatus

router = APIRouter(prefix="/stream", tags=["Stream"])

# Registry of active camera statuses (populated by pipeline)
_camera_registry: Dict[str, CameraStatus] = {}

# Active WebSocket connections: camera_id → list of websockets
_connections: Dict[str, list] = {}


def register_camera(status: CameraStatus):
    """Called by the pipeline to register a camera."""
    _camera_registry[status.camera_id] = status
    if status.camera_id not in _connections:
        _connections[status.camera_id] = []


def get_registry():
    return _camera_registry


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.get("/cameras")
def list_cameras():
    """Return all registered cameras and their current status."""
    return [
        {
            "camera_id":   s.camera_id,
            "name":        s.name,
            "connected":   s.connected,
            "fps":         s.fps,
            "frame_count": s.frame_count,
            "last_error":  s.last_error,
        }
        for s in _camera_registry.values()
    ]


@router.get("/status")
def stream_status():
    """Return overall stream health."""
    total    = len(_camera_registry)
    online   = sum(1 for s in _camera_registry.values() if s.connected)
    return {
        "total_cameras":  total,
        "online_cameras": online,
        "offline_cameras": total - online,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── WebSocket: live frame stream ──────────────────────────────────────────────

@router.websocket("/ws/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """
    Stream live annotated frames to a WebSocket client as base64-encoded JPEG.

    Message format (JSON):
    {
        "type":      "frame",
        "camera_id": "cam_0",
        "timestamp": "2026-11-15 09:00:00",
        "fps":       5.0,
        "frame":     "<base64-encoded JPEG>"
    }
    """
    await websocket.accept()
    print(f"[Stream] 🔌 WebSocket connected → camera '{camera_id}'")

    if camera_id not in _connections:
        _connections[camera_id] = []
    _connections[camera_id].append(websocket)

    try:
        while True:
            # Keep the connection alive — frames are pushed by broadcast_frame()
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"[Stream] 🔌 WebSocket disconnected → camera '{camera_id}'")
    finally:
        if websocket in _connections.get(camera_id, []):
            _connections[camera_id].remove(websocket)


async def broadcast_frame(camera_id: str, frame, fps: float = 0.0):
    """
    Encode a frame as JPEG and broadcast it to all WebSocket clients
    watching the given camera. Called from the pipeline thread via asyncio.
    """
    clients = _connections.get(camera_id, [])
    if not clients:
        return

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64       = base64.b64encode(buffer).decode("utf-8")

    message = json.dumps({
        "type":      "frame",
        "camera_id": camera_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fps":       fps,
        "frame":     b64,
    })

    dead = []
    for ws in clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.remove(ws)


async def broadcast_event(camera_id: str, event):
    """
    Push a new event notification to all WebSocket clients watching a camera.
    """
    clients = _connections.get(camera_id, [])
    if not clients:
        return

    message = json.dumps({
        "type":      "event",
        "camera_id": camera_id,
        "id":        event.id,
        "timestamp": event.timestamp,
        "severity":  event.severity,
        "label":     event.label,
        "category":  event.category,
    })

    dead = []
    for ws in clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.remove(ws)
