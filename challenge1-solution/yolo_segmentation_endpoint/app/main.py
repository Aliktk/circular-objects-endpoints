# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import os
import shutil
import json

from .database import get_db_connection
from .models import (
    ImageUploadResponse,
    ImageSegmentationResponse,
    SegmentationObject,
    SegmentationMask,
    Centroid
)
from .utils import yolo_segmentation_model
from .logger import logger

app = FastAPI(title="YOLO Detection and Segmentation API")

# Ensure storage/uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created upload directory at {UPLOAD_DIR}.")

# Optional: Ensure storage/masks directory exists if you choose to store masks as files
MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "masks")
if not os.path.exists(MASKS_DIR):
    os.makedirs(MASKS_DIR)
    logger.info(f"Created masks directory at {MASKS_DIR}.")

# Existing Detection Endpoints
# ---------------------------------
# If you have existing detection endpoints, include them here.
# For example:
#
# @app.post("/detect-image", response_model=ImageUploadResponse, summary="Upload an image for detection.")
# async def detect_image(file: UploadFile = File(...)):
#     # Your detection endpoint implementation
#     pass
#
# ---------------------------------

# New Segmentation Endpoints

@app.post("/segment-image", response_model=ImageUploadResponse, summary="Upload an image for segmentation.")
async def segment_image(file: UploadFile = File(...)):
    logger.info(f"Received segmentation upload request for file: {file.filename}")
    try:
        # Validate file type
        allowed_extensions = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"}
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in allowed_extensions:
            logger.warning(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        # Save the uploaded image to the upload directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded image to {file_path}.")
        
        # Insert image record into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (filename) VALUES (?)", (file.filename,))
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Inserted image record into database with ID: {image_id}.")
        
        # Check if the model is loaded
        if yolo_segmentation_model is None:
            logger.error("YOLO Segmentation model is not initialized.")
            raise HTTPException(status_code=500, detail="Segmentation model not available.")
        
        # Process the image with YOLO segmentation model
        detected_segments = yolo_segmentation_model.detect_segments(file_path, target_classes=["coin"])
        
        # Insert detected segmentation objects into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        for segment in detected_segments:
            # Convert mask data to JSON string or handle as per your storage strategy
            segmentation_mask_json = json.dumps(segment["segmentation_mask"])
            centroid_json = json.dumps(segment["centroid"])
            radius = segment["radius"]
            area = segment["area"]
            cursor.execute("""
                INSERT INTO segmentation_objects (image_id, segmentation_mask, centroid, radius, area)
                VALUES (?, ?, ?, ?, ?)
            """, (image_id, segmentation_mask_json, centroid_json, radius, area))
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(detected_segments)} segmentation objects into the database for image ID: {image_id}.")
        
        return ImageUploadResponse(image_id=image_id, filename=file.filename)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in segment_image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/images/{image_id}/segmentation", response_model=ImageSegmentationResponse, summary="Retrieve segmentation results for a given image.")
def get_segmentation_results(image_id: int):
    logger.info(f"Received request to get segmentation results for image ID: {image_id}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Verify image exists
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            image = cursor.fetchone()
            if not image:
                logger.error(f"Image with ID {image_id} not found in database.")
                raise HTTPException(status_code=404, detail="Image not found")
            else:
                logger.debug(f"Image found: {{'id': {image['id']}, 'filename': '{image['filename']}', 'upload_time': '{image['upload_time']}'}}")

            # Retrieve segmentation objects
            cursor.execute("""
                SELECT id, segmentation_mask, centroid, radius, area FROM segmentation_objects
                WHERE image_id = ?
            """, (image_id,))
            rows = cursor.fetchall()
            logger.debug(f"Retrieved {len(rows)} segmentation objects from the database for image ID: {image_id}")

        segmentation_objects = []
        for row in rows:
            try:
                segmentation_mask = json.loads(row["segmentation_mask"])
                centroid = Centroid(**json.loads(row["centroid"]))
                radius = row["radius"]
                area = row["area"]
                segmentation_objects.append(SegmentationObject(
                    id=row["id"],
                    segmentation_mask=SegmentationMask(mask_data=segmentation_mask),
                    centroid=centroid,
                    radius=radius,
                    area=area
                ))
                logger.debug(f"Processed segmentation object: ID={row['id']}, Mask={segmentation_mask}, Centroid={centroid}, Radius={radius}, Area={area}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for segmentation object ID {row['id']}: {str(e)}")

        logger.info(f"Retrieved {len(segmentation_objects)} segmentation objects for image ID: {image_id}.")
        return ImageSegmentationResponse(image_id=image_id, segmentation_objects=segmentation_objects)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_segmentation_results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/circular-objects/{object_id}/radius", response_model=Dict[str, float], summary="Retrieve the radius of a specific circular object.")
def get_circular_object_radius(object_id: int):
    logger.info(f"Received request to get radius for circular object ID: {object_id}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Retrieve the circular object
            cursor.execute("""
                SELECT id, radius FROM segmentation_objects
                WHERE id = ?
            """, (object_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Circular object with ID {object_id} not found in database.")
                raise HTTPException(status_code=404, detail="Circular object not found")
            else:
                radius = row["radius"]
                logger.debug(f"Circular object found: ID={row['id']}, Radius={radius}")

        # Prepare and return the response
        return {"id": row["id"], "radius": radius}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_circular_object_radius: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

