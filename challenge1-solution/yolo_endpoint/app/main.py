# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from typing import List
import os
import shutil
import json

from .database import get_db_connection
from .models import (
    ImageUploadResponse,
    ImageCircularObjectsResponse,
    CircularObjectDetail,
    CircularObject,
    CircularObjectRadiusResponse)
from .utils import yolo_model
from .logger import logger

app = FastAPI(title="Yellow Detection API")

# Ensure storage/uploads directory exists
UPLOAD_DIR = "storage/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created upload directory at {UPLOAD_DIR}.")

@app.post("/upload-image", response_model=ImageUploadResponse, summary="Upload an image for processing.")
async def upload_image(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    try:
        # Save the uploaded image to the upload directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded image to {file_path}.")

        # Begin database transaction
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO images (filename) VALUES (?)", (file.filename,))
            image_id = cursor.lastrowid
            logger.info(f"Inserted image record into database with ID: {image_id}.")

            # Process the image with YOLO to detect circular objects
            detected_objects = yolo_model.detect_circles(file_path, target_classes=["coin"])

            # Insert detected circular objects into the database
            for obj in detected_objects:
                bounding_box_json = json.dumps(obj["bounding_box"])
                centroid_json = json.dumps(obj["centroid"])
                radius = obj["radius"]
                cursor.execute("""
                    INSERT INTO circular_objects (image_id, bounding_box, centroid, radius)
                    VALUES (?, ?, ?, ?)
                """, (image_id, bounding_box_json, centroid_json, radius))
            conn.commit()
            logger.info(f"Inserted {len(detected_objects)} circular objects into the database for image ID: {image_id}.")

        # Retrieve the inserted circular objects with their IDs
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, bounding_box FROM circular_objects
                WHERE image_id = ?
            """, (image_id,))
            rows = cursor.fetchall()

        # Prepare response with CircularObject instances
        circular_objects = []
        for row in rows:
            bounding_box = json.loads(row["bounding_box"])
            circular_objects.append(CircularObject(id=row["id"], bounding_box=bounding_box))
            logger.debug(f"Processed circular object: ID={row['id']}, Bounding Box={bounding_box}")

        return ImageUploadResponse(image_id=image_id, filename=file.filename)
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/images/{image_id}/circular-objects", response_model=ImageCircularObjectsResponse, summary="Retrieve all circular objects for a given image.")
def get_circular_objects(image_id: int):
    logger.info(f"Received request to get circular objects for image ID: {image_id}")
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
                logger.debug(f"Image found: {dict(image)}")

            # Retrieve circular objects
            cursor.execute("""
                SELECT id, bounding_box FROM circular_objects
                WHERE image_id = ?
            """, (image_id,))
            rows = cursor.fetchall()
            logger.debug(f"Retrieved {len(rows)} circular objects from the database for image ID: {image_id}")

        circular_objects = []
        for row in rows:
            try:
                bounding_box = json.loads(row["bounding_box"])
                circular_objects.append(CircularObject(id=row["id"], bounding_box=bounding_box))
                logger.debug(f"Processed circular object: ID={row['id']}, Bounding Box={bounding_box}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for circular object ID {row['id']}: {str(e)}")

        logger.info(f"Retrieved {len(circular_objects)} circular objects for image ID: {image_id}.")
        return ImageCircularObjectsResponse(image_id=image_id, circular_objects=circular_objects)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_circular_objects: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
@app.get("/circular-objects/{object_id}/radius", response_model=CircularObjectRadiusResponse, summary="Retrieve the radius of a specific circular object.")
def get_circular_object_radius(object_id: int):
    logger.info(f"Received request to get radius for circular object ID: {object_id}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Retrieve the circular object
            cursor.execute("""
                SELECT id, radius FROM circular_objects
                WHERE id = ?
            """, (object_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Circular object with ID {object_id} not found in database.")
                raise HTTPException(status_code=404, detail="Circular object not found")
            else:
                logger.debug(f"Circular object found: ID={row['id']}, Radius={row['radius']}")

        # Prepare and return the response
        return CircularObjectRadiusResponse(id=row["id"], radius=row["radius"])
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_circular_object_radius: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")