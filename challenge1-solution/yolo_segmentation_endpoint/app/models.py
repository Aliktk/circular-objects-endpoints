# app/models.py

from pydantic import BaseModel
from typing import List

class SegmentationMask(BaseModel):
    mask_data: List[List[float]]  # Changed from str to list of lists

class Centroid(BaseModel):
    x: float
    y: float

class SegmentationObject(BaseModel):
    id: int
    segmentation_mask: SegmentationMask
    centroid: Centroid
    radius: float
    area: float  # Added area field

class ImageSegmentationResponse(BaseModel):
    image_id: int
    segmentation_objects: List[SegmentationObject]

class ImageUploadResponse(BaseModel):
    image_id: int
    filename: str
