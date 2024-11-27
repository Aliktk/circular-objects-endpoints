# app/utils.py

import os
import shutil
from uuid import uuid4
from fastapi import UploadFile
from typing import List, Dict, Optional
from .logger import logger
import cv2

def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Save uploaded file to the destination directory with a unique filename.
    Returns the filename.
    """
    try:
        _, ext = os.path.splitext(upload_file.filename)
        unique_filename = f"{uuid4()}{ext}"
        file_path = os.path.join(destination, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return unique_filename
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def create_overlay_image(image_path: str, circles: List[Dict], ground_truth: Optional[List[Dict]] = None) -> str:
    """
    Create an image overlaying detected circles and ground truth circles.
    Saves the overlay image to a temporary file and returns the file path.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Image file '{image_path}' not found.")
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        logger.info(f"Creating overlay image for: {image_path}")
        
        overlay_img = img.copy()
        
        # Draw detected circles in green
        for circle in circles:
            x, y = circle['centroid_x'], circle['centroid_y']
            r = circle['radius']
            cv2.circle(overlay_img, (x, y), r, (0, 255, 0), 2)  # Green
            cv2.circle(overlay_img, (x, y), 2, (0, 0, 255), 3)  # Red center
            logger.debug(f"Drew detected circle: ID={circle['id']}, Centroid=({x},{y}), Radius={r}")
        
        # Draw ground truth circles in blue
        if ground_truth:
            for gt in ground_truth:
                x, y = gt['centroid_x'], gt['centroid_y']
                r = gt['radius']
                cv2.circle(overlay_img, (x, y), r, (255, 0, 0), 2)  # Blue
                cv2.circle(overlay_img, (x, y), 2, (0, 0, 255), 3)  # Red center
                logger.debug(f"Drew ground truth circle: Centroid=({x},{y}), Radius={r}")
        
        # Save the overlay image
        overlay_filename = f"overlay_{uuid4()}.jpg"
        overlay_path = os.path.join("uploads", overlay_filename)
        cv2.imwrite(overlay_path, overlay_img)
        logger.info(f"Overlay image created: {overlay_path}")
        
        return overlay_path
    except Exception as e:
        logger.error(f"Error creating overlay image: {e}")
        raise
