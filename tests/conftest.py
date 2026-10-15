"""Shared test setup. Runs before any test file is imported."""
import io
import os

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

os.environ.setdefault("VERIFY_MODEL", "v1")   # small model whose weights are in the repo: fast tests


def make_png(source_type: str | None = None) -> bytes:
    """A tiny PNG, optionally carrying an IPTC DigitalSourceType label in its XMP metadata."""
    info = PngInfo()
    if source_type:
        xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
               f'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/{source_type}"/>'
               '</rdf:RDF></x:xmpmeta>')
        info.add_itxt("XML:com.adobe.xmp", xmp)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 40)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def make_jpeg(quality: int, with_exif: bool = False) -> bytes:
    img = Image.new("RGB", (64, 64), (30, 90, 160))
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
