# app/models.py

from pydantic import BaseModel
from typing import List

class CircularObjectRadiusResponse(BaseModel):
    id: int
    radius: float

class ImageUploadResponse(BaseModel):
    image_id: int
    filename: str

class CircularObject(BaseModel):
    id: int
    bounding_box: dict

class CircularObjectDetail(BaseModel):
    id: int
    bounding_box: dict
    centroid: dict
    radius: float

class ImageCircularObjectsResponse(BaseModel):
    image_id: int
    circular_objects: List[CircularObject]



