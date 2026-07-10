"""
history_store.py — a local SQLite log of the single-node legacy feed's raw
telemetry fields (see data_feed.py's single-node get_rack_id()/poll()
API), one row per replayed sample.

This is pure storage: every field written here already flows through
data_feed.poll(). Nothing in this module computes a score, verdict, or
alert — see README "Known gaps" for why that distinction matters in this
repo.
"""
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rack_history.db")

_conn = [None]


def init_history_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rack_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rack_id TEXT NOT NULL,
            sample_index INTEGER,
            captured_at REAL NOT NULL,
            frq_hz REAL,
            total_power_w REAL,
            average_gpu_temp_c REAL,
            status TEXT
        )
    """)
    conn.commit()
    _conn[0] = conn


def record_sample(rack_id: str, data: dict):
    conn = _conn[0]
    conn.execute(
        """INSERT INTO rack_history
           (rack_id, sample_index, captured_at, frq_hz, total_power_w,
            average_gpu_temp_c, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            rack_id,
            data.get("index"),
            time.time(),
            data.get("frq_hz"),
            data.get("total_power_w"),
            data.get("average_gpu_temp_c"),
            str(data.get("status", "--")),
        ),
    )
    conn.commit()


def get_recent(rack_id: str, limit: int = 50) -> list[dict]:
    conn = _conn[0]
    rows = conn.execute(
        """SELECT sample_index, captured_at, frq_hz, total_power_w,
                  average_gpu_temp_c, status
           FROM rack_history
           WHERE rack_id = ?
           ORDER BY id DESC
           LIMIT ?""",
        (rack_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]
