"""
main.py: the front desk of Verify.
Receives requests, validates them, asks model.py for a verdict, stores it via db.py,
and replies with JSON. No machine learning and no SQL live in this file.
"""
import hashlib
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from app import db
from app.model import MODEL_VERSION, load_image, predict

MAX_BYTES = 10 * 1024 * 1024   # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()               # runs once when the server starts
    yield


app = FastAPI(
    title="Verify",
    description="Detects AI-generated images, built for content forwarded on WhatsApp.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------- Response shapes ----------
class ScanResult(BaseModel):
    scan_id: int
    created_at: str
    verdict: Literal["likely_ai", "inconclusive", "likely_real"]
    p_fake: float
    latency_ms: int
    model_version: str
    image_sha256: str
    cached: bool


class Stats(BaseModel):
    total_scans: int
    by_verdict: dict[str, int]
    avg_latency_ms: float | None
    model_version: str


class ErrorMessage(BaseModel):
    detail: str


def _to_result(row: dict, cached: bool) -> ScanResult:
    """Database row -> API reply. The DB calls it 'id'; the API calls it 'scan_id'."""
    return ScanResult(
        scan_id=row["id"], created_at=row["created_at"], verdict=row["verdict"],
        p_fake=row["p_fake"], latency_ms=row["latency_ms"],
        model_version=row["model_version"], image_sha256=row["image_sha256"], cached=cached,
    )


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post(
    "/scans",
    response_model=ScanResult,
    status_code=201,
    responses={
        200: {"model": ScanResult, "description": "Cached: this exact image was already scanned by this model"},
        413: {"model": ErrorMessage, "description": "File larger than 10 MB"},
        415: {"model": ErrorMessage, "description": "File is not a valid image"},
    },
)
def create_scan(response: Response, file: UploadFile = File(...)):
    # 1. Size check: read at most 10 MB + 1 byte
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    # 2. Fingerprint, then the cheap cache lookup (before any expensive work)
    sha256 = hashlib.sha256(data).hexdigest()
    previous = db.find_by_hash(sha256, MODEL_VERSION)
    if previous:
        response.status_code = 200          # nothing new created
        return _to_result(previous, cached=True)

    # 3. Is it really an image?
    try:
        img = load_image(data)
    except ValueError:
        raise HTTPException(status_code=415, detail="File is not a valid image.")

    # 4. The expensive part, then store it (201 Created)
    result = predict(img)
    row = db.insert_scan(sha256, result)
    return _to_result(row, cached=False)


@app.get("/scans", response_model=list[ScanResult])
def list_recent_scans(limit: int = Query(20, ge=1, le=100)):
    return [_to_result(r, cached=False) for r in db.list_scans(limit)]


@app.get(
    "/scans/{scan_id}",
    response_model=ScanResult,
    responses={404: {"model": ErrorMessage, "description": "No scan with this ID"}},
)
def read_scan(scan_id: int):
    row = db.get_scan(scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")
    return _to_result(row, cached=False)


@app.get("/stats", response_model=Stats)
def stats():
    return Stats(**db.get_stats(), model_version=MODEL_VERSION)