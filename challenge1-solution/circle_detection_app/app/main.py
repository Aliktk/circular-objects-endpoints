# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
from typing import List, Optional
import sqlite3
from .database import (
    create_tables,
    insert_image,
    insert_circle,
    get_circles_by_image,
    get_circle_by_id,
    get_image_filename,
    get_all_images
)
from .circle_detector import CircleDetector
from .utils import save_upload_file, create_overlay_image
import csv
import numpy as np
from .logger import logger
from .models import (
    UploadImageResponse,
    CircleResponse,
    EvaluationMetrics
)

app = FastAPI(title="Circle Detection API")

# Initialize directories
UPLOAD_DIR = "uploads"
GROUND_TRUTH_DIR = "ground_truth"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

# Initialize database tables on startup
@app.on_event("startup")
def startup_event():
    logger.info("Starting up and creating tables if they do not exist.")
    try:
        create_tables()
        logger.info("Database tables are set up successfully.")
    except Exception as e:
        logger.critical(f"Critical error during startup: {e}")
        raise

# Initialize CircleDetector
circle_detector = CircleDetector()

@app.post("/upload_image", response_model=UploadImageResponse)
async def upload_image(image: UploadFile = File(...)):
    """
    Upload an image, detect circles, and store data.
    """
    logger.info("Received request to upload image.")
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            logger.warning(f"Uploaded file is not an image: {image.content_type}")
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        
        # Save the uploaded image
        filename = save_upload_file(image, UPLOAD_DIR)
        image_id = str(uuid.uuid4())
        logger.debug(f"Generated Image ID: {image_id}")
        
        try:
            insert_image(image_id, filename)
            logger.info(f"Inserted image into database: ID={image_id}, Filename={filename}")
        except sqlite3.IntegrityError:
            logger.error("Image with this filename already exists.")
            raise HTTPException(status_code=400, detail="Image with this filename already exists.")
        except Exception as e:
            logger.error(f"Error inserting image into database: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
        # Detect circles
        image_path = os.path.join(UPLOAD_DIR, filename)
        try:
            circles = circle_detector.detect_circles(image_path, image_id)
            logger.info(f"Detected {len(circles)} circles in image ID={image_id}")
        except FileNotFoundError as e:
            logger.error(e)
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error during circle detection: {e}")
            raise HTTPException(status_code=500, detail="Error during circle detection.")
        
        # Insert circles into the database
        inserted_circles = []
        for circle in circles:
            try:
                insert_circle(circle)
                # Convert to CircleResponse model
                inserted_circles.append(CircleResponse(**circle))
                logger.debug(f"Inserted circle into database: ID={circle['id']}")
            except sqlite3.IntegrityError:
                logger.error(f"Circle with ID {circle['id']} already exists.")
                continue  # Skip inserting this circle
            except Exception as e:
                logger.error(f"Error inserting circle into database: {e}")
                continue  # Skip inserting this circle
        
        logger.info(f"Image uploaded and processed successfully: {image_id}")
        
        return UploadImageResponse(
            image_id=image_id,
            filename=filename,
            detected_circles=inserted_circles
        )
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during image upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/images/{image_id}/circles", response_model=List[CircleResponse])
async def list_circles(image_id: str):
    """
    Retrieve all circles for a given image (id and bounding box).
    """
    logger.info(f"Received request to list circles for Image ID={image_id}")
    try:
        # Check if image exists
        filename = get_image_filename(image_id)
        if not filename:
            logger.warning(f"Image ID={image_id} not found.")
            raise HTTPException(status_code=404, detail="Image not found.")
        
        # Get circles
        circles = get_circles_by_image(image_id)
        logger.info(f"Retrieved {len(circles)} circles for Image ID={image_id}")
        # Convert to CircleResponse models
        circles_response = [CircleResponse(**circle) for circle in circles]
        return circles_response
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while listing circles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/circles/{circle_id}", response_model=CircleResponse)
async def get_circle_details(circle_id: str):
    """
    Retrieve properties of a specific circle.
    """
    logger.info(f"Received request to get details for Circle ID={circle_id}")
    try:
        circle = get_circle_by_id(circle_id)
        if not circle:
            logger.warning(f"Circle ID={circle_id} not found.")
            raise HTTPException(status_code=404, detail="Circle not found.")
        logger.info(f"Retrieved details for Circle ID={circle_id}")
        return CircleResponse(**circle)
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while retrieving circle details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/download_image_with_mask/{image_id}", response_class=FileResponse)
async def download_image_with_mask(image_id: str, ground_truth_file: Optional[UploadFile] = File(None)):
    """
    Retrieve the original image overlaid with detected circles and optionally ground truth circles.
    If ground_truth_file is provided, it should be a CSV with columns: centroid_x, centroid_y, radius
    """
    logger.info(f"Received request to download image with mask for Image ID={image_id}")
    try:
        # Check if image exists
        filename = get_image_filename(image_id)
        if not filename:
            logger.warning(f"Image ID={image_id} not found.")
            raise HTTPException(status_code=404, detail="Image not found.")
        
        image_path = os.path.join(UPLOAD_DIR, filename)
        
        # Get detected circles
        circles = get_circles_by_image(image_id)
        logger.debug(f"Detected {len(circles)} circles for overlay.")
        
        # Process ground truth if provided
        ground_truth = []
        if ground_truth_file:
            logger.info("Processing ground truth file for overlay.")
            # Validate ground truth file type
            if not ground_truth_file.content_type.startswith("text/csv"):
                logger.warning(f"Ground truth file is not a CSV: {ground_truth_file.content_type}")
                raise HTTPException(status_code=400, detail="Ground truth file must be a CSV.")
            
            # Save the ground truth file
            gt_filename = save_upload_file(ground_truth_file, GROUND_TRUTH_DIR)
            gt_path = os.path.join(GROUND_TRUTH_DIR, gt_filename)
            logger.debug(f"Saved ground truth file: {gt_path}")
            
            # Read ground truth circles
            with open(gt_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        gt_circle = {
                            'centroid_x': int(row['centroid_x']),
                            'centroid_y': int(row['centroid_y']),
                            'radius': int(row['radius'])
                        }
                        ground_truth.append(gt_circle)
                        logger.debug(f"Loaded ground truth circle: {gt_circle}")
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error parsing ground truth row {row}: {e}")
                        continue
        
        # Create overlay image
        try:
            overlay_path = create_overlay_image(image_path, circles, ground_truth if ground_truth else None)
            logger.info(f"Overlay image created: {overlay_path}")
        except FileNotFoundError as e:
            logger.error(e)
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error creating overlay image: {e}")
            raise HTTPException(status_code=500, detail="Error creating overlay image.")
        
        return FileResponse(overlay_path, media_type="image/jpeg", filename=os.path.basename(overlay_path))
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating overlay image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/evaluate_model", response_model=EvaluationMetrics)
async def evaluate_model(ground_truth_file: UploadFile = File(...)):
    """
    Evaluate the circle detection model against ground truth data.
    Ground truth file should be a CSV with columns: image_id, centroid_x, centroid_y, radius
    """
    logger.info("Received request to evaluate model with ground truth data.")
    try:
        # Validate ground truth file type
        if not ground_truth_file.content_type.startswith("text/csv"):
            logger.warning(f"Ground truth file is not a CSV: {ground_truth_file.content_type}")
            raise HTTPException(status_code=400, detail="Ground truth file must be a CSV.")
        
        # Save the ground truth file
        gt_filename = save_upload_file(ground_truth_file, GROUND_TRUTH_DIR)
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_filename)
        logger.debug(f"Saved ground truth file: {gt_path}")
        
        # Load ground truth data
        ground_truth = {}
        with open(gt_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('image_id')
                try:
                    centroid_x = int(row['centroid_x'])
                    centroid_y = int(row['centroid_y'])
                    radius = int(row['radius'])
                except (ValueError, KeyError) as e:
                    logger.error(f"Error parsing ground truth row {row}: {e}")
                    continue
                if image_id not in ground_truth:
                    ground_truth[image_id] = []
                ground_truth[image_id].append({
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'radius': radius
                })
        logger.info(f"Loaded ground truth data for {len(ground_truth)} images.")
        
        # Initialize metrics
        total_tp = total_fp = total_fn = 0
        
        # Iterate through ground truth data
        for image_id, gt_circles in ground_truth.items():
            logger.debug(f"Evaluating Image ID={image_id} with {len(gt_circles)} ground truth circles.")
            # Get detected circles for the image
            detected_circles = get_circles_by_image(image_id)
            logger.debug(f"Found {len(detected_circles)} detected circles for Image ID={image_id}")
            
            matched_gt = set()
            matched_det = set()
            
            for gt in gt_circles:
                for det in detected_circles:
                    if det['id'] in matched_det:
                        continue
                    distance = np.sqrt((gt['centroid_x'] - det['centroid_x'])**2 + (gt['centroid_y'] - det['centroid_y'])**2)
                    radius_diff = abs(gt['radius'] - det['radius'])
                    if distance <= 10 and radius_diff <= 10:
                        total_tp += 1
                        matched_det.add(det['id'])
                        logger.debug(
                            f"Matched Ground Truth Circle with Detected Circle: "
                            f"GT=({gt['centroid_x']},{gt['centroid_y']}), R={gt['radius']} | "
                            f"Detected ID={det['id']}, Centroid=({det['centroid_x']},{det['centroid_y']}), R={det['radius']}"
                        )
                        break
            total_fp += len(detected_circles) - len(matched_det)
            total_fn += len(gt_circles) - total_tp
            logger.debug(f"Image ID={image_id}: TP={total_tp}, FP={len(detected_circles) - len(matched_det)}, FN={len(gt_circles) - total_tp}")
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        
        logger.info(f"Evaluation Metrics: Precision={precision}, Recall={recall}, F1-Score={f1_score}")
        
        metrics = EvaluationMetrics(
            Precision=precision,
            Recall=recall,
            F1_Score=f1_score  # Note: Changed to valid Python identifier
        )
        
        return metrics
    except Exception as e:
        logger.error(f"Unexpected error while evaluating model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")