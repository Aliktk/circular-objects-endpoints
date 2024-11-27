# app/schemas.py

from pydantic import BaseModel

class ImageResponse(BaseModel):
    depth: float
    image: str  # Base64 encoded image

    class Config:
        orm_mode = True
