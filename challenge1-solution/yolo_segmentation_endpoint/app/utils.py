# app/utils.py

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict
from ultralytics import YOLO  # Ensure ultralytics is installed
from .logger import logger

class YOLOSegmentationModel:
    def __init__(self, model_path: str):
        """
        Initializes the YOLO segmentation model.

        Args:
            model_path (str): Path to the pre-trained YOLO segmentation model (.pt file).
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO Segmentation model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO Segmentation model: {str(e)}")
            raise e

    def detect_segments(self, image_path: str, target_classes: List[str] = ["coin"], conf_threshold: float = 0.25) -> List[Dict]:
        """
        Performs segmentation on the provided image and returns segmentation data.

        Args:
            image_path (str): Path to the image file.
            target_classes (List[str], optional): List of target class names to detect. Defaults to ["coin"].
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.25.

        Returns:
            List[Dict]: List of detected segmentation objects with mask, centroid, radius, and area.
        """
        logger.info(f"Performing segmentation on image: {image_path}")
        detected_segments = []
        
        try:
            # Perform inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Iterate over detections
            for result in results:
                for detection in result.boxes.data.tolist():
                    # detection format: [x1, y1, x2, y2, confidence, class]
                    x1, y1, x2, y2, conf, cls = detection
                    class_name = self.model.names[int(cls)]
                    
                    if class_name not in target_classes:
                        continue  # Skip non-target classes
                    
                    # Extract mask if available
                    if result.masks is not None and len(result.masks.data) > 0:
                        mask = result.masks.data[0].cpu().numpy()  # Assuming single mask per detection
                        mask_json = mask.tolist()  # Convert to list for JSON serialization
                    else:
                        mask_json = []  # Empty mask
                    
                    # Calculate centroid
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroid = {"x": centroid_x, "y": centroid_y}
                    
                    # Calculate radius (assuming circular coins)
                    width = x2 - x1
                    height = y2 - y1
                    radius = (width + height) / 4  # Average radius
                    
                    # Calculate area of the circle
                    area = np.pi * (radius ** 2)
                    
                    detected_segments.append({
                        "segmentation_mask": mask_json,  # Could also store mask as base64 or file path
                        "centroid": centroid,
                        "radius": radius,
                        "area": area
                    })
            
            logger.info(f"Detected {len(detected_segments)} segmentation objects in image: {image_path}")
        except Exception as e:
            logger.error(f"Error during segmentation inference: {str(e)}")
        
        return detected_segments

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_model", "best.pt")
# Instantiate the YOLOSegmentationModel with the path to your pre-trained model
try:
    yolo_segmentation_model = YOLOSegmentationModel(model_path=MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize YOLOSegmentationModel: {str(e)}")
    yolo_segmentation_model = None
