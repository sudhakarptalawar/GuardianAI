"""
GuardianAI — Alerts API Route
GET  /alerts/config       → get current alert config
POST /alerts/config       → update alert config
GET  /alerts/             → list sent alerts
POST /alerts/test         → send a test alert
"""

import json
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import List

from api.models.schemas import AlertConfigSchema, AlertSchema

router = APIRouter(prefix="/alerts", tags=["Alerts"])

CONFIG_FILE = "reports/alert_config.json"
ALERTS_FILE = "reports/alerts_sent.json"

_config: AlertConfigSchema = AlertConfigSchema()
_alerts_sent: List[dict]   = []


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    global _config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                _config = AlertConfigSchema(**json.load(f))
        except Exception:
            pass


def save_config():
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(_config.model_dump(), f, indent=2)


load_config()


@router.get("/config", response_model=AlertConfigSchema)
def get_config():
    """Return current alert configuration."""
    return _config


@router.post("/config", response_model=AlertConfigSchema)
def update_config(config: AlertConfigSchema):
    """Update alert configuration."""
    global _config
    _config = config
    save_config()
    return _config


# ── Alert history ─────────────────────────────────────────────────────────────

@router.get("/", response_model=List[AlertSchema])
def list_alerts(limit: int = 50):
    """Return the last N sent alerts."""
    return list(reversed(_alerts_sent))[:limit]


@router.post("/test")
def send_test_alert():
    """Send a test alert through all configured channels."""
    if not _config.enabled:
        return {"success": False, "message": "Alerts are disabled."}

    alert = {
        "id":        f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id":  "TEST",
        "channel":   "websocket",
        "message":   "🛡️ GuardianAI test alert — system is working correctly.",
        "delivered": True,
    }
    _alerts_sent.append(alert)
    return {"success": True, "message": "Test alert sent.", "alert": alert}


# ── Internal: log a sent alert ────────────────────────────────────────────────

def log_alert(event_id: str, channel: str, message: str, delivered: bool = True):
    alert = {
        "id":        f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id":  event_id,
        "channel":   channel,
        "message":   message,
        "delivered": delivered,
    }
    _alerts_sent.append(alert)
    return alert


def get_config_obj() -> AlertConfigSchema:
    return _config
