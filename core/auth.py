"""
GuardianAI – Auth Module
JWT token creation/verification + bcrypt password hashing.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY  = os.environ.get("GUARDIAN_SECRET", "guardian-ai-secret-key-change-in-production-2024")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 7   # 7 days

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password helpers ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


# ── JWT helpers ──────────────────────────────────────────────────────────────

def create_token(user_id: int, email: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "role": role, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── User DB helpers (uses same SQLite as rest of app) ────────────────────────

def _conn(db_path: str = "reports/guardianai.db"):
    c = sqlite3.connect(db_path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def ensure_users_table(db_path: str = "reports/guardianai.db"):
    with _conn(db_path) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                email      TEXT    NOT NULL UNIQUE,
                password   TEXT    NOT NULL,
                role       TEXT    NOT NULL DEFAULT 'operator',
                org        TEXT    NOT NULL DEFAULT '',
                avatar_col TEXT    NOT NULL DEFAULT '#4ce0c0',
                created_at TEXT    NOT NULL
            );
        """)

def create_user(name: str, email: str, password: str, role: str = "operator", org: str = "", avatar_col: str = "#4ce0c0", db_path: str = "reports/guardianai.db") -> Optional[dict]:
    try:
        hashed = hash_password(password)
        now    = datetime.now(timezone.utc).isoformat()
        with _conn(db_path) as c:
            cur = c.execute(
                "INSERT INTO users (name, email, password, role, org, avatar_col, created_at) VALUES (?,?,?,?,?,?,?)",
                (name, email, hashed, role, org, avatar_col, now)
            )
            return get_user_by_id(cur.lastrowid, db_path)
    except sqlite3.IntegrityError:
        return None   # email already exists

def get_user_by_email(email: str, db_path: str = "reports/guardianai.db") -> Optional[dict]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: int, db_path: str = "reports/guardianai.db") -> Optional[dict]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

def update_user(user_id: int, fields: dict, db_path: str = "reports/guardianai.db") -> Optional[dict]:
    allowed = {"name", "org", "avatar_col"}
    sets    = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return get_user_by_id(user_id, db_path)
    sql  = "UPDATE users SET " + ", ".join(f"{k}=?" for k in sets) + " WHERE id=?"
    vals = list(sets.values()) + [user_id]
    with _conn(db_path) as c:
        c.execute(sql, vals)
    return get_user_by_id(user_id, db_path)

def change_password(user_id: int, new_password: str, db_path: str = "reports/guardianai.db") -> bool:
    hashed = hash_password(new_password)
    with _conn(db_path) as c:
        c.execute("UPDATE users SET password=? WHERE id=?", (hashed, user_id))
    return True

def authenticate_user(email: str, password: str, db_path: str = "reports/guardianai.db") -> Optional[dict]:
    user = get_user_by_email(email, db_path)
    if user and verify_password(password, user["password"]):
        return user
    return None
