"""
fingerprint.py: perceptual hashing, so a forwarded copy can be matched to an image seen before.
SHA-256 changes completely when one byte changes; a perceptual hash (pHash) describes what the
picture looks like, so recompression and resizing barely change it.

Measured on 33 original/WhatsApp-forward pairs: same image <= 2 bits apart (of 64),
different images >= 18 bits apart (1,876 pairs). MAX_DISTANCE = 8 sits safely between.
"""
import imagehash
from PIL import Image

MAX_DISTANCE = 8


def phash(img: Image.Image) -> str:
    """64-bit perceptual hash as 16 hex characters."""
    return str(imagehash.phash(img))


def distance(a: str, b: str) -> int:
    """Hamming distance: how many of the 64 bits differ."""
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
