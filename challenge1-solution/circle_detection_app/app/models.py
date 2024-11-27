# app/models.py

from pydantic import BaseModel
from typing import List

class CircleBase(BaseModel):
    id: str
    centroid_x: int
    centroid_y: int
    radius: int
    bounding_box_x: int
    bounding_box_y: int
    bounding_box_width: int
    bounding_box_height: int

class CircleCreate(CircleBase):
    image_id: str

class CircleResponse(CircleBase):
    pass

class UploadImageResponse(BaseModel):
    image_id: str
    filename: str
    detected_circles: List[CircleResponse]

class EvaluationMetrics(BaseModel):
    Precision: float
    Recall: float
    F1_Score: float  # Changed to valid Python identifier
