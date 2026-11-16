"""
GuardianAI — Alert Manager
Delivers alerts via email (SMTP) and in-app WebSocket push.
SMS via Twilio is scaffolded — enable in Week 3.
"""

import smtplib
import threading
import json
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from core.severity import Severity


ALERT_LOG = "reports/alerts_sent.json"
CONFIG_FILE = "reports/alert_config.json"

# Severity order for threshold comparison
SEVERITY_ORDER = {
    Severity.LOW.value:      1,
    Severity.MEDIUM.value:   2,
    Severity.HIGH.value:     3,
    Severity.CRITICAL.value: 4,
}


class AlertManager:
    """
    Routes detection events to configured alert channels.

    Config keys:
        enabled       : bool — master on/off
        min_severity  : str  — minimum severity to alert (LOW/MEDIUM/HIGH/CRITICAL)
        email         : str  — recipient email address
        smtp_host     : str  — SMTP host (default: smtp.gmail.com)
        smtp_port     : int  — SMTP port (default: 587)
        smtp_user     : str  — SMTP username
        smtp_pass     : str  — SMTP password
        phone         : str  — phone number for SMS (Twilio, Week 3)
    """

    def __init__(self, config: Optional[dict] = None):
        self._config   = config or self._load_config()
        self._log: list = self._load_log()
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_alert(self, severity: str) -> bool:
        if not self._config.get("enabled", True):
            return False
        min_sev = self._config.get("min_severity", "HIGH")
        return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(min_sev, 3)

    def send(self, event) -> list[str]:
        """
        Evaluate an event and send alerts via all configured channels.
        Returns list of channels notified.
        """
        if not self.should_alert(event.severity):
            return []

        channels = []

        # Email
        if self._config.get("email"):
            sent = self._send_email(event)
            if sent:
                channels.append("email")

        # WebSocket push (handled by stream router — just log here)
        channels.append("websocket")

        # Log
        self._log_alert(event, channels)
        return channels

    def update_config(self, config: dict):
        self._config = config
        self._save_config(config)

    def get_config(self) -> dict:
        return dict(self._config)

    def get_log(self, limit: int = 50) -> list:
        return list(reversed(self._log))[:limit]

    # ── Email ─────────────────────────────────────────────────────────────────

    def _send_email(self, event) -> bool:
        try:
            recipient = self._config["email"]
            smtp_host = self._config.get("smtp_host", "smtp.gmail.com")
            smtp_port = int(self._config.get("smtp_port", 587))
            smtp_user = self._config.get("smtp_user", "")
            smtp_pass = self._config.get("smtp_pass", "")

            if not smtp_user or not smtp_pass:
                print("[AlertManager] Email credentials not configured — skipping email.")
                return False

            severity_emoji = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
            }.get(event.severity, "⚪")

            subject = f"{severity_emoji} GuardianAI Alert: {event.label}"
            body    = f"""GuardianAI Security Alert
{'=' * 40}
{event.label}

Time      : {event.timestamp}
Camera    : {event.camera_id}
Severity  : {event.severity}
Category  : {event.category}
Confidence: {round(event.confidence * 100)}%

{event.description}

{'=' * 40}
GuardianAI Intelligent Safety System
"""

            msg = MIMEMultipart()
            msg["From"]    = smtp_user
            msg["To"]      = recipient
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient, msg.as_string())

            print(f"[AlertManager] 📧 Email sent → {recipient} ({event.severity})")
            return True

        except Exception as ex:
            print(f"[AlertManager] Email failed: {ex}")
            return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def _log_alert(self, event, channels: list):
        entry = {
            "id":        f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_id":  event.id,
            "severity":  event.severity,
            "label":     event.label,
            "channels":  channels,
        }
        with self._lock:
            self._log.append(entry)
            self._save_log()

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"enabled": True, "min_severity": "HIGH"}

    def _save_config(self, config: dict):
        os.makedirs("reports", exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

    def _load_log(self) -> list:
        if os.path.exists(ALERT_LOG):
            try:
                with open(ALERT_LOG) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_log(self):
        try:
            with open(ALERT_LOG, "w") as f:
                json.dump(self._log[-500:], f, indent=2)   # keep last 500
        except Exception:
            pass
