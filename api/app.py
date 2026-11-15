"""
GuardianAI — FastAPI Application
Main API server that powers the GuardianAI dashboard.

Run with:
    python -m uvicorn api.app:app --reload --port 8000

Endpoints:
    GET  /                      → system info
    GET  /health                → health check
    GET  /events/               → list events (filterable)
    GET  /events/summary        → severity summary
    GET  /events/{id}           → single event
    GET  /alerts/config         → alert config
    POST /alerts/config         → update alert config
    POST /alerts/test           → send test alert
    GET  /stream/cameras        → camera status list
    GET  /stream/status         → stream health
    WS   /stream/ws/{camera_id} → live frame + event WebSocket
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.event_manager   import EventManager
from api.routes           import events, alerts, stream
from api.routes.events    import set_event_manager

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "GuardianAI API",
    description = "Intelligent Safety System — REST + WebSocket API",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (allow React frontend on any localhost port) ─────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────

event_manager = EventManager()
set_event_manager(event_manager)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(events.router)
app.include_router(alerts.router)
app.include_router(stream.router)

# ── Root endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system":  "GuardianAI",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    summary = event_manager.summary()
    return {
        "status":       "ok",
        "total_events": summary["total_events"],
        "cameras":      len(stream.get_registry()),
    }
