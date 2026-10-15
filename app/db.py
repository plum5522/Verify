"""
db.py: everything about storing scans. SQLite, one file, one table.
main.py calls these functions and never writes SQL itself.
"""
import json
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
    """Create the table and index if missing, and migrate older tables. Safe to run every startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                image_sha256  TEXT    NOT NULL,
                p_fake        REAL    NOT NULL,
                verdict       TEXT    NOT NULL,
                model_version TEXT    NOT NULL,
                latency_ms    INTEGER NOT NULL,
                signals       TEXT    NOT NULL DEFAULT '[]',
                phash         TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hash_model ON scans (image_sha256, model_version)")

        # Migration 1: tables created before the provenance layer have no 'signals' column.
        # ALTER TABLE adds it in place; existing rows get the default '[]' and nothing is lost.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
        if "signals" not in columns:
            conn.execute("ALTER TABLE scans ADD COLUMN signals TEXT NOT NULL DEFAULT '[]'")
        # Migration 2: perceptual hash for matching forwarded copies (NULL for older rows)
        if "phash" not in columns:
            conn.execute("ALTER TABLE scans ADD COLUMN phash TEXT")


def _row(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d["signals"] = json.loads(d["signals"])  # stored as JSON text, returned as a list
    return d


def insert_scan(image_sha256: str, result: dict) -> dict:
    """Save one scan and return the stored row (including its new id)."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scans (created_at, image_sha256, p_fake, verdict, model_version, latency_ms, "
            "signals, phash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (created_at, image_sha256, result["p_fake"], result["verdict"], result["model_version"],
             result["latency_ms"], json.dumps(result["signals"]), result.get("phash")),
        )
        new_id = cur.lastrowid
    return get_scan(new_id)


def get_scan(scan_id: int) -> dict | None:
    with _connect() as conn:
        return _row(conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone())


def find_by_hash(image_sha256: str, model_version: str) -> dict | None:
    """Has this exact image already been judged by this exact pipeline version?"""
    with _connect() as conn:
        return _row(conn.execute(
            "SELECT * FROM scans WHERE image_sha256 = ? AND model_version = ? ORDER BY id DESC LIMIT 1",
            (image_sha256, model_version),
        ).fetchone())


def declared_ai_hashes() -> list[tuple[int, str]]:
    """(id, phash) of every scan whose file carried an AI label.
    Compared in Python: SQLite has no bit-count function. A linear scan is fine for thousands
    of rows; at millions this would need an index for Hamming distance (e.g. a BK-tree)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, phash FROM scans WHERE verdict = 'ai_declared' AND phash IS NOT NULL").fetchall()
    return [(r["id"], r["phash"]) for r in rows]


def list_scans(limit: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


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
