"""
GuardianAI — SQLite Database Layer
Replaces the JSON flat-file store with a proper SQLite database.
Handles events, cameras, and alert config with full CRUD.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict


DB_PATH = "reports/guardianai.db"


@dataclass
class EventRecord:
    id:          str
    timestamp:   str
    camera_id:   str
    category:    str
    severity:    str
    label:       str
    description: str
    confidence:  float
    clip_path:   Optional[str] = None


class Database:
    """
    SQLite-backed storage for GuardianAI events, cameras, and config.
    Thread-safe via check_same_thread=False + WAL mode.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _migrate(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                camera_id   TEXT NOT NULL DEFAULT 'cam_0',
                category    TEXT NOT NULL,
                severity    TEXT NOT NULL,
                label       TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                confidence  REAL NOT NULL DEFAULT 0.0,
                clip_path   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_severity  ON events(severity);
            CREATE INDEX IF NOT EXISTS idx_events_camera    ON events(camera_id);

            CREATE TABLE IF NOT EXISTS cameras (
                camera_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                location    TEXT NOT NULL DEFAULT 'Unknown',
                source      TEXT NOT NULL DEFAULT '0',
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alert_config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── Events ────────────────────────────────────────────────────────────────

    def insert_event(self, event: EventRecord) -> bool:
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO events
                   (id, timestamp, camera_id, category, severity, label, description, confidence, clip_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.id, event.timestamp, event.camera_id, event.category,
                 event.severity, event.label, event.description,
                 event.confidence, event.clip_path),
            )
            self._conn.commit()
            return True
        except Exception as ex:
            print(f"[DB] insert_event error: {ex}")
            return False

    def get_events(
        self,
        limit:     int           = 100,
        severity:  Optional[str] = None,
        camera_id: Optional[str] = None,
        search:    Optional[str] = None,
        days:      Optional[int] = None,
    ) -> List[EventRecord]:
        clauses = []
        params  = []

        if severity:
            clauses.append("severity = ?")
            params.append(severity.upper())
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if search:
            clauses.append("(label LIKE ? OR category LIKE ? OR description LIKE ? OR camera_id LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term, term])
        if days:
            cutoff = (datetime.now() - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            clauses.append("timestamp >= ?")
            params.append(cutoff)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

        return [EventRecord(**dict(r)) for r in rows]

    def get_event_by_id(self, event_id: str) -> Optional[EventRecord]:
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return EventRecord(**dict(row)) if row else None

    def count_events(self, severity: Optional[str] = None, days: Optional[int] = None) -> int:
        clauses, params = [], []
        if severity:
            clauses.append("severity = ?")
            params.append(severity.upper())
        if days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(f"SELECT COUNT(*) FROM events {where}", params).fetchone()
        return row[0]

    def clear_events(self):
        self._conn.execute("DELETE FROM events")
        self._conn.commit()

    def daily_counts(self, days: int = 7) -> List[dict]:
        from datetime import timedelta
        result = []
        for i in range(days - 1, -1, -1):
            day   = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            count = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp LIKE ?",
                (f"{day}%",),
            ).fetchone()[0]
            result.append({"date": day, "count": count})
        return result

    def category_counts(self, days: Optional[int] = None) -> dict:
        if days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            rows = self._conn.execute(
                "SELECT category, COUNT(*) as cnt FROM events WHERE timestamp >= ? GROUP BY category ORDER BY cnt DESC",
                (cutoff,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT category, COUNT(*) as cnt FROM events GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
        return {r["category"]: r["cnt"] for r in rows}

    def severity_counts(self, days: Optional[int] = None) -> dict:
        if days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            rows = self._conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM events WHERE timestamp >= ? GROUP BY severity",
                (cutoff,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM events GROUP BY severity"
            ).fetchall()
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for r in rows:
            counts[r["severity"]] = r["cnt"]
        return counts

    # ── Alert config ──────────────────────────────────────────────────────────

    def get_alert_config(self) -> dict:
        rows = self._conn.execute("SELECT key, value FROM alert_config").fetchall()
        config = {"enabled": True, "min_severity": "HIGH", "email": None, "phone": None}
        for r in rows:
            try:
                config[r["key"]] = json.loads(r["value"])
            except Exception:
                config[r["key"]] = r["value"]
        return config

    def set_alert_config(self, config: dict):
        for key, value in config.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO alert_config (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        self._conn.commit()

    def close(self):
        self._conn.close()


# ── Singleton ─────────────────────────────────────────────────────────────────

_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
