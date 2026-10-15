"""API tests: real FastAPI app, real v1 model, temporary database."""
from tests.conftest import make_jpeg, make_png


def upload(client, data, name="img.png"):
    return client.post("/scans", files={"file": (name, data, "application/octet-stream")})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_declared_ai_image_gets_ai_declared_verdict(client):
    r = upload(client, make_png("trainedAlgorithmicMedia"))
    assert r.status_code == 201
    body = r.json()
    assert body["verdict"] == "ai_declared"
    assert {s["source"] for s in body["signals"]} >= {"iptc", "model"}


def test_unlabelled_image_is_never_cleared_as_real(client):
    verdict = upload(client, make_jpeg(quality=80), "fwd.jpg").json()["verdict"]
    assert verdict in {"likely_ai", "inconclusive"}


def test_repeat_upload_is_cached_with_same_signals(client):
    data = make_png("trainedAlgorithmicMedia")
    first = upload(client, data).json()
    second = upload(client, data)
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["signals"] == first["signals"]
    assert second.json()["scan_id"] == first["scan_id"]


def test_not_an_image_is_415(client):
    assert upload(client, b"hello, not an image", "x.png").status_code == 415


def test_too_large_is_413(client):
    assert upload(client, b"0" * (10 * 1024 * 1024 + 1), "big.png").status_code == 413


def test_missing_file_is_422(client):
    assert client.post("/scans").status_code == 422


def test_unknown_scan_is_404(client):
    assert client.get("/scans/99999").status_code == 404


def test_limit_over_100_is_rejected(client):
    assert client.get("/scans?limit=101").status_code == 422


def test_migration_adds_signals_column(tmp_path, monkeypatch):
    """An old database without the 'signals' column is upgraded in place, keeping its rows."""
    import sqlite3
    from app import db
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
                 "image_sha256 TEXT NOT NULL, p_fake REAL NOT NULL, verdict TEXT NOT NULL, "
                 "model_version TEXT NOT NULL, latency_ms INTEGER NOT NULL)")
    conn.execute("INSERT INTO scans VALUES (1, 't', 'abc', 0.5, 'inconclusive', 'old', 10)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    row = db.get_scan(1)
    assert row["signals"] == [] and row["verdict"] == "inconclusive"
