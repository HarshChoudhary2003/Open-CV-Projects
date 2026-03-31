"""
database/db_manager.py
SQLite event-log persistence layer.
"""

import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                severity    TEXT    NOT NULL DEFAULT 'INFO',
                description TEXT,
                snapshot    TEXT,
                track_id    INTEGER
            );

            CREATE TABLE IF NOT EXISTS known_persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                added_at    TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        """)
        conn.commit()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
def log_event(event_type: str, description: str, severity: str = "INFO",
              snapshot: Optional[str] = None, track_id: Optional[int] = None) -> None:
    """Insert one event row."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO events (timestamp, event_type, severity, description, snapshot, track_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, event_type, severity, description, snapshot, track_id)
        )
        conn.commit()
        conn.close()


def fetch_recent_events(limit: int = 200,
                        severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent events as list of dicts."""
    with _lock:
        conn = _get_conn()
        if severity_filter and severity_filter != "ALL":
            rows = conn.execute(
                "SELECT * FROM events WHERE severity=? ORDER BY id DESC LIMIT ?",
                (severity_filter, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def fetch_event_stats() -> Dict[str, Any]:
    """Aggregate counts for dashboard KPIs."""
    with _lock:
        conn = _get_conn()
        total     = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        alerts    = conn.execute("SELECT COUNT(*) FROM events WHERE severity='ALERT'").fetchone()[0]
        warnings  = conn.execute("SELECT COUNT(*) FROM events WHERE severity='WARNING'").fetchone()[0]
        today_str = datetime.now().strftime("%Y-%m-%d")
        today     = conn.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp LIKE ?", (f"{today_str}%",)
        ).fetchone()[0]
        conn.close()
    return {"total": total, "alerts": alerts, "warnings": warnings, "today": today}


def clear_old_events(days: int = 30) -> None:
    """Prune events older than `days` days."""
    cutoff = datetime.now().strftime(f"%Y-%m-%d")  # simple daily prune point
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM events WHERE timestamp < date(?, ?)", (cutoff, f"-{days} days"))
        conn.commit()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    log_event("SYSTEM", "Database initialised", severity="INFO")
    print("DB ready →", DB_PATH)
