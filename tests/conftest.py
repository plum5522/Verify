"""Shared test setup. Runs before any test file is imported."""
import io
import os

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

os.environ.setdefault("VERIFY_MODEL", "v1")   # small model whose weights are in the repo: fast tests


def pattern(seed: int = 0) -> Image.Image:
    """A 256x256 picture of random coloured blocks. Different seeds give different pictures.
    (Plain one-colour images would all look identical to a perceptual hash.)"""
    import random
    rng = random.Random(seed)
    small = Image.new("RGB", (8, 8))
    small.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(64)])
    return small.resize((256, 256), Image.BICUBIC)


def make_png(source_type: str | None = None, seed: int = 0) -> bytes:
    """A tiny PNG, optionally carrying an IPTC DigitalSourceType label in its XMP metadata."""
    info = PngInfo()
    if source_type:
        xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
               f'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/{source_type}"/>'
               '</rdf:RDF></x:xmpmeta>')
        info.add_itxt("XML:com.adobe.xmp", xmp)
    buf = io.BytesIO()
    pattern(seed).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def make_jpeg(quality: int, with_exif: bool = False, seed: int = 1) -> bytes:
    img = pattern(seed)
    buf = io.BytesIO()
    if with_exif:
        exif = Image.Exif()
        exif[0x0110] = "iPhone 15"            # camera model tag
        img.save(buf, "JPEG", quality=quality, exif=exif)
    else:
        img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real API, but with a throwaway database in a temporary folder."""
    from fastapi.testclient import TestClient
    from app import db, main
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    with TestClient(main.app) as c:          # 'with' runs startup, so the table gets created
        yield c
