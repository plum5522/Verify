"""Unit tests for provenance.check: labels in, findings out. No model, no server."""
from app import provenance
from tests.conftest import make_jpeg, make_png


def findings(data):
    return [(s["source"], s["finding"]) for s in provenance.check(data)["signals"]]


def test_declared_ai_label_is_detected():
    r = provenance.check(make_png("trainedAlgorithmicMedia"))
    assert r["ai_declared"] is True
    assert ("iptc", "ai_generated") in findings(make_png("trainedAlgorithmicMedia"))


def test_ai_edited_photo_is_not_called_ai_generated():
    r = provenance.check(make_png("compositeWithTrainedAlgorithmicMedia"))
    assert r["ai_declared"] is False
    assert ("iptc", "ai_edited") in findings(make_png("compositeWithTrainedAlgorithmicMedia"))


def test_c2pa_style_packed_value_is_detected():
    # Inside C2PA (binary CBOR) the value runs straight into the next field name
    data = make_png() + b"jumb c2pa digitalsourcetype/trainedAlgorithmicMediakdescription OpenAI"
    r = provenance.check(data)
    assert r["ai_declared"] is True
    assert ("c2pa", "present") in findings(data)


def test_camera_photo_label_is_not_ai():
    r = provenance.check(make_png("digitalCapture"))
    assert r["ai_declared"] is False


def test_whatsapp_like_jpeg_is_flagged_as_stripped():
    detail = provenance.check(make_jpeg(quality=80))["signals"][0]["detail"]
    assert "WhatsApp" in detail


def test_other_jpeg_quality_is_not_called_whatsapp():
    detail = provenance.check(make_jpeg(quality=95))["signals"][0]["detail"]
    assert "WhatsApp" not in detail


def test_photo_with_exif_but_no_label():
    assert findings(make_jpeg(quality=95, with_exif=True)) == [("metadata", "no_ai_label")]


def test_garbage_bytes_do_not_crash():
    assert provenance.check(b"not an image")["ai_declared"] is False
