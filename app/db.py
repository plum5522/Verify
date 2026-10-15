"""
db.py: everything about storing scans. SQLite, one file, one table.
main.py calls these functions and never writes SQL itself.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "verify.db"   # verify/verify.db (gitignored)


@contextmanager
def _connect():
    """Open a connection, commit if all went well, always close. One per call."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts: row["verdict"]
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the table and index if they don't exist yet. Safe to run every startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                image_sha256  TEXT    NOT NULL,
                p_fake        REAL    NOT NULL,
                verdict       TEXT    NOT NULL,
                model_version TEXT    NOT NULL,
                latency_ms    INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hash_model ON scans (image_sha256, model_version)")


def insert_scan(image_sha256: str, result: dict) -> dict:
    """Save one scan and return the stored row (including its new id)."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scans (created_at, image_sha256, p_fake, verdict, model_version, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, image_sha256, result["p_fake"], result["verdict"],
             result["model_version"], result["latency_ms"]),
        )
        new_id = cur.lastrowid
    return get_scan(new_id)


def get_scan(scan_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return dict(row) if row else None


def find_by_hash(image_sha256: str, model_version: str) -> dict | None:
    """Has this exact image already been judged by this exact model?"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE image_sha256 = ? AND model_version = ? ORDER BY id DESC LIMIT 1",
            (image_sha256, model_version),
        ).fetchone()
    return dict(row) if row else None


def list_scans(limit: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        by_verdict = {r["verdict"]: r["n"] for r in conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM scans GROUP BY verdict")}
        avg = conn.execute("SELECT AVG(latency_ms) FROM scans").fetchone()[0]
    return {
        "total_scans": total,
        "by_verdict": by_verdict,
        "avg_latency_ms": round(avg, 1) if avg is not None else None,
    }