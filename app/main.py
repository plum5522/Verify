"""
main.py: the front desk of Verify.
Receives requests, validates them, asks model.py for a verdict, replies with JSON.
No machine learning lives in this file.
"""
import hashlib
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.model import MODEL_VERSION, load_image, predict

MAX_BYTES = 10 * 1024 * 1024   # 10 MB

app = FastAPI(
    title="Verify",
    description="Detects AI-generated images, built for content forwarded on WhatsApp.",
    version="0.1.0",
)


class ScanResult(BaseModel):
    """The exact shape of every successful /scans reply."""
    verdict: Literal["likely_ai", "inconclusive", "likely_real"]
    p_fake: float
    latency_ms: int
    model_version: str
    image_sha256: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/scans", response_model=ScanResult)
def create_scan(file: UploadFile = File(...)):
    # 1. Size check first: read at most 10 MB + 1 byte, never the whole huge file
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    # 2. Is it really an image? (rotation fix + RGB happen inside load_image)
    try:
        img = load_image(data)
    except ValueError:
        raise HTTPException(status_code=415, detail="File is not a valid image.")

    # 3. Fingerprint (will be used by the database step)
    sha256 = hashlib.sha256(data).hexdigest()

    # 4. The expensive part, last
    result = predict(img)
    return ScanResult(**result, image_sha256=sha256)