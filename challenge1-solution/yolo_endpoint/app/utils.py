# app/utils.py

import os
from ultralytics import YOLO
from .logger import logger
from typing import List, Dict

class YOLOModel:
    def __init__(self, model_path: str):
        """
        Initializes the YOLOv8 model.

        Args:
            model_path (str): The file path to the trained YOLOv8 .pt model.
        """
        if not os.path.exists(model_path):
            logger.error(f"YOLOv8 model file not found at {model_path}")
            raise FileNotFoundError(f"YOLOv8 model file not found at {model_path}")
        
        try:
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully.")
            logger.info(f"Model Classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {str(e)}")
            raise e

    def detect_circles(self, image_path: str, target_classes: List[str] = ["coin", "circle"]) -> List[Dict]:
        """
        Detects specified objects in the given image using the YOLOv8 model.

        Args:
            image_path (str): The file path to the image for detection.
            target_classes (List[str], optional): Classes to filter detections. Defaults to ["coin", "circle"].

        Returns:
            List[Dict]: A list of detected objects with their bounding boxes, centroids, and radii.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found at {image_path}")
            raise FileNotFoundError(f"Image file not found at {image_path}")
        
        try:
            # Perform inference on the image
            results = self.model(image_path, verbose=False)
            detected_objects = []

            # Iterate through each detected object in the results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls)
                    class_name = self.model.names[cls_id]
                    confidence = box.conf.item()
                    logger.info(f"Detected class: {class_name} with confidence: {confidence:.2f}")

                    # Filter detections to include only target classes
                    if class_name.lower() not in [cls.lower() for cls in target_classes]:
                        continue

                    # Extract bounding box coordinates
                    # Adjusted to handle nested lists
                    xyxy = box.xyxy.tolist()
                    if len(xyxy) == 0:
                        logger.warning("Empty bounding box detected.")
                        continue
                    if isinstance(xyxy[0], list) or isinstance(xyxy[0], tuple):
                        # Access the first list within the nested list
                        x_min, y_min, x_max, y_max = xyxy[0]
                    else:
                        # If it's not nested, unpack directly
                        x_min, y_min, x_max, y_max = xyxy

                    # Calculate bounding box dimensions
                    width = x_max - x_min
                    height = y_max - y_min

                    # Calculate centroid coordinates
                    centroid = {
                        "x": (x_min + x_max) / 2,
                        "y": (y_min + y_max) / 2
                    }

                    # Estimate radius as the average of half the width and half the height
                    radius = (width / 2 + height / 2) / 2

                    # Create bounding box dictionary
                    bounding_box = {
                        "x": int(x_min),
                        "y": int(y_min),
                        "width": int(width),
                        "height": int(height)
                    }

                    # Append the detected object's details to the list
                    detected_objects.append({
                        "class": class_name,
                        "bounding_box": bounding_box,
                        "centroid": centroid,
                        "radius": float(radius)
                    })

            logger.info(f"Detected {len(detected_objects)} objects: {', '.join([obj['class'] for obj in detected_objects])}")
            return detected_objects

        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            raise e

# Initialize YOLO model with the actual path to your YOLOv8 .pt file
# Replace the path below with the path to your trained YOLOv8 model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_model", "best.pt")

try:
    yolo_segmentation_model = YOLOModel(model_path=MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize YOLOSegmentationModel: {str(e)}")
    yolo_segmentation_model = None
