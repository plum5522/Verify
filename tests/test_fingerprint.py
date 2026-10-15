"""The perceptual hash must survive WhatsApp-style recompression but separate different pictures."""
import io

from PIL import Image

from app import fingerprint
from tests.conftest import pattern


def recompress(img, size, quality=80):
    buf = io.BytesIO()
    img.resize(size).save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def test_recompressed_copy_stays_close():
    a = fingerprint.phash(pattern(3))
    b = fingerprint.phash(recompress(pattern(3), (180, 180)))
    assert fingerprint.distance(a, b) <= fingerprint.MAX_DISTANCE


def test_different_pictures_are_far_apart():
    hashes = [fingerprint.phash(pattern(s)) for s in range(20)]
    closest = min(fingerprint.distance(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1:])
    assert closest > fingerprint.MAX_DISTANCE
