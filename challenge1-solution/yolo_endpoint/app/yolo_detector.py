# app/yolo_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import List, Dict
from app.logger import logger

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the YOLOv8 model.
        """
        if not os.path.exists(model_path):
            logger.info(f"Downloading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)

    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        Detect objects in the image and return their details.
        """
        results = self.model(image_path)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = self.model.names[class_id]

                # Assuming circular objects are labeled appropriately
                if label.lower() in ['circle', 'coin', 'disk']:
                    obj = {
                        "bounding_box": {
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        },
                        "confidence": confidence
                    }
                    # Calculate centroid and radius
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    radius = (x2 - x1) / 2  # Approximation
                    obj["centroid"] = {"x": centroid_x, "y": centroid_y}
                    obj["radius"] = radius

                    detections.append(obj)
                    logger.info(f"Detected {label} with confidence {confidence:.2f} at ({centroid_x:.2f}, {centroid_y:.2f}) with radius {radius:.2f}")
        return detections
