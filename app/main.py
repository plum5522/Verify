"""
main.py: the front desk of Verify.
Receives requests, validates them, gathers evidence (provenance labels, then the model),
decides a verdict, stores it via db.py, and replies with JSON.
No machine learning and no SQL live in this file.
"""
import hashlib
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from app import db, provenance
from app.model import MODEL_VERSION, load_image, predict

MAX_BYTES = 10 * 1024 * 1024   # 10 MB
# Cache key = model AND decision rules. Bumping "prov1" invalidates answers made under old rules.
ANALYSIS_VERSION = f"{MODEL_VERSION}+prov1"

Verdict = Literal["ai_declared", "likely_ai", "inconclusive", "likely_real"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()               # runs once when the server starts (creates or migrates the table)
    yield


app = FastAPI(
    title="Verify",
    description="Checks whether an image is AI-generated, built for content forwarded on WhatsApp. "
                "Reports its evidence and says 'inconclusive' when the evidence is weak.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------- Response shapes ----------
class Signal(BaseModel):
    source: str        # "iptc", "c2pa", "metadata", "model"
    finding: str
    detail: str


class ScanResult(BaseModel):
    scan_id: int
    created_at: str
    verdict: Verdict
    p_fake: float
    latency_ms: int
    model_version: str
    image_sha256: str
    cached: bool
    signals: list[Signal]


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
        p_fake=row["p_fake"], latency_ms=row["latency_ms"], model_version=row["model_version"],
        image_sha256=row["image_sha256"], cached=cached, signals=row["signals"],
    )


def decide(prov: dict, model_result: dict) -> tuple[str, list[dict]]:
    """Combine the evidence into one verdict.
    1. A declared AI label wins: the generator itself says so.
    2. The model may raise suspicion (likely_ai) but never clears an image: on the external test,
       every model's "real" answers were wrong for most AI images, so 'likely_real' becomes 'inconclusive'.
    """
    model_signal = {"source": "model", "finding": model_result["verdict"],
                    "detail": f"{MODEL_VERSION}: p_fake {model_result['p_fake']:.3f}. "
                              "Pixel-based detection is unreliable on current generators and forwarded images."}
    signals = prov["signals"] + [model_signal]
    if prov["ai_declared"]:
        return "ai_declared", signals
    if model_result["verdict"] == "likely_ai":
        return "likely_ai", signals
    return "inconclusive", signals


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "analysis_version": ANALYSIS_VERSION}


@app.post(
    "/scans",
    response_model=ScanResult,
    status_code=201,
    responses={
        200: {"model": ScanResult, "description": "Cached: this exact image was already analysed"},
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
    previous = db.find_by_hash(sha256, ANALYSIS_VERSION)
    if previous:
        response.status_code = 200          # nothing new created
        return _to_result(previous, cached=True)

    # 3. Is it really an image?
    try:
        img = load_image(data)
    except ValueError:
        raise HTTPException(status_code=415, detail="File is not a valid image.")

    # 4. Evidence: labels in the raw bytes (decoding threw the metadata away), then the model
    prov = provenance.check(data)
    result = predict(img)
    result["verdict"], result["signals"] = decide(prov, result)
    result["model_version"] = ANALYSIS_VERSION

    # 5. Store it (201 Created)
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
    return Stats(**db.get_stats(), model_version=ANALYSIS_VERSION)
