"""
provenance.py: what does the file say about itself?
Runs on the raw uploaded bytes, BEFORE decoding, because decoding to pixels discards all metadata.
No machine learning here: these are labels written by the generator (or stripped by an app).

Signals:
  IPTC DigitalSourceType "trainedAlgorithmicMedia"              -> the creator declares it AI-generated
  IPTC DigitalSourceType "compositeWithTrainedAlgorithmicMedia" -> a photo edited with AI
  C2PA manifest present  -> Content Credentials exist (cameras sign real photos too, so not proof of AI)
  no EXIF + JPEG quality-80 tables -> consistent with a WhatsApp forward (labels probably stripped)

Limitation: labels are DETECTED, not cryptographically VERIFIED. A label can be forged or removed.
"""
import io
import re

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()                               # so iPhone HEIC metadata can be read too

_DST = re.compile(rb"digitalsourcetype/([A-Za-z]+)", re.IGNORECASE)   # IPTC vocabulary URI
_AI_GENERATED = b"trainedalgorithmicmedia"
_AI_EDITED = b"compositewithtrainedalgorithmicmedia"
_ISSUERS = [b"OpenAI", b"Google", b"Microsoft", b"Adobe", b"Meta"]


def _quality80_tables() -> dict:
    """Pillow's standard JPEG tables at quality 80: WhatsApp's exact setting (measured)."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "JPEG", quality=80, subsampling=2)
    return {k: list(v) for k, v in Image.open(io.BytesIO(buf.getvalue())).quantization.items()}


_WHATSAPP_TABLES = _quality80_tables()


def check(data: bytes) -> dict:
    """Return {"ai_declared": bool, "signals": [{"source", "finding", "detail"}, ...]}."""
    signals = []
    low = data.lower()

    # 1. IPTC DigitalSourceType (in XMP, and also inside C2PA manifests).
    #    Inside C2PA the value is packed against the next field name (binary CBOR),
    #    e.g. "trainedAlgorithmicMediakdescription", so match the start of the word.
    values = {m.lower() for m in _DST.findall(data)}
    ai_declared = any(v.startswith(_AI_GENERATED) for v in values)
    ai_edited = any(v.startswith(_AI_EDITED) for v in values)
    if ai_declared:
        signals.append({"source": "iptc", "finding": "ai_generated",
                        "detail": "File declares DigitalSourceType trainedAlgorithmicMedia"})
    elif ai_edited:
        signals.append({"source": "iptc", "finding": "ai_edited",
                        "detail": "File declares it was edited with generative AI"})

    # 2. C2PA Content Credentials (JUMBF container labelled c2pa)
    if b"c2pa" in low and b"jumb" in low:
        issuers = [n.decode() for n in _ISSUERS if n.lower() in low]
        signals.append({"source": "c2pa", "finding": "present",
                        "detail": "Content Credentials found" +
                                  (f" (mentions {', '.join(issuers)})" if issuers else "") +
                                  "; signature not verified"})

    # 3. Missing metadata: the usual state of a forwarded image
    try:
        img = Image.open(io.BytesIO(data))           # reads headers only, no pixel decoding
        exif_tags = len(img.getexif())
        tables = getattr(img, "quantization", None)
        is_jpeg = img.format == "JPEG"
    except Exception:
        exif_tags, tables, is_jpeg = 0, None, False

    if not signals and exif_tags == 0:
        if is_jpeg and tables and {k: list(v) for k, v in tables.items()} == _WHATSAPP_TABLES:
            detail = ("No metadata, and JPEG settings match WhatsApp's re-encoding (quality 80): "
                      "likely a forwarded copy, so any AI label would have been removed")
        else:
            detail = "No provenance labels and no camera metadata: absence proves nothing either way"
        signals.append({"source": "metadata", "finding": "stripped", "detail": detail})
    elif not signals:
        signals.append({"source": "metadata", "finding": "no_ai_label",
                        "detail": f"{exif_tags} EXIF tags, no AI label: not proof the image is real"})

    return {"ai_declared": ai_declared, "signals": signals}
