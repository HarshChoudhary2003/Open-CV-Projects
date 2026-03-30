"""
database/db.py
--------------
SQLite persistence layer for retail analytics events.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager


DB_PATH = Path(__file__).parent.parent / "data" / "retail.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT NOT NULL,
                source      TEXT
            );

            CREATE TABLE IF NOT EXISTS footfall_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                timestamp   TEXT NOT NULL,
                hour        TEXT NOT NULL,
                track_id    INTEGER NOT NULL,
                UNIQUE(session_id, track_id)
            );

            CREATE TABLE IF NOT EXISTS dwell_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                track_id    INTEGER NOT NULL,
                dwell_sec   REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS heatmap_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                timestamp   TEXT NOT NULL,
                hot_zones   TEXT          -- JSON array
            );

            CREATE TABLE IF NOT EXISTS frame_metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                timestamp   TEXT NOT NULL,
                frame_no    INTEGER NOT NULL,
                active_ids  INTEGER NOT NULL,
                total_ids   INTEGER NOT NULL,
                avg_dwell   REAL NOT NULL
            );
        """)
    print(f"[DB] Initialized → {DB_PATH}")


# ── Session management ─────────────────────────────────────────────────────
def create_session(source: str = "webcam") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (started_at, source) VALUES (?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), source)
        )
        return cur.lastrowid


# ── Write helpers ──────────────────────────────────────────────────────────
def log_footfall(session_id: int, track_ids: list[int]):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    hour = time.strftime("%H:00")
    with get_db() as conn:
        for tid in track_ids:
            conn.execute(
                "INSERT OR IGNORE INTO footfall_log (session_id, timestamp, hour, track_id) "
                "VALUES (?, ?, ?, ?)",
                (session_id, now, hour, tid)
            )


def log_dwell(session_id: int, dwell_times: dict[int, float]):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        for tid, secs in dwell_times.items():
            conn.execute(
                "INSERT INTO dwell_events (session_id, track_id, dwell_sec, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, tid, secs, now)
            )


def log_heatmap(session_id: int, hot_zones: list[dict]):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO heatmap_snapshots (session_id, timestamp, hot_zones) "
            "VALUES (?, ?, ?)",
            (session_id, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(hot_zones))
        )


def log_frame_metrics(session_id: int, frame_no: int,
                      active_ids: int, total_ids: int, avg_dwell: float):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO frame_metrics (session_id, timestamp, frame_no, active_ids, total_ids, avg_dwell) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, time.strftime("%Y-%m-%d %H:%M:%S"),
             frame_no, active_ids, total_ids, avg_dwell)
        )


# ── Read helpers ───────────────────────────────────────────────────────────
def get_all_sessions() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT s.id, s.started_at, s.source, COUNT(DISTINCT f.track_id) as visitors "
            "FROM sessions s "
            "LEFT JOIN footfall_log f ON f.session_id = s.id "
            "GROUP BY s.id ORDER BY s.started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_hourly_footfall(session_id: int | None = None) -> list[dict]:
    with get_db() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT hour, COUNT(*) as count FROM footfall_log "
                "WHERE session_id=? GROUP BY hour ORDER BY hour",
                (session_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT hour, COUNT(*) as count FROM footfall_log "
                "GROUP BY hour ORDER BY hour"
            ).fetchall()
        return [dict(r) for r in rows]


def get_dwell_stats(session_id: int | None = None) -> dict:
    with get_db() as conn:
        q = "SELECT dwell_sec FROM dwell_events"
        args = ()
        if session_id:
            q += " WHERE session_id=?"
            args = (session_id,)
        rows = conn.execute(q, args).fetchall()
        dwells = [r["dwell_sec"] for r in rows]
        if not dwells:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "total": 0}
        import statistics
        return {
            "avg": statistics.mean(dwells),
            "max": max(dwells),
            "min": min(dwells),
            "total": len(dwells),
        }


def get_frame_metrics(session_id: int | None = None, limit: int = 500) -> list[dict]:
    with get_db() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM frame_metrics WHERE session_id=? ORDER BY frame_no DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM frame_metrics ORDER BY frame_no DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
