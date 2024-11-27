# app/models.py

from dataclasses import dataclass

@dataclass
class ImageData:
    depth: float
    image: bytes
