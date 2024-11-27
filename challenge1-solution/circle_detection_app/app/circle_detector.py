# app/circle_detector.py

import cv2
import numpy as np
import uuid
from typing import List, Dict
from .logger import logger

class CircleDetector:
    def __init__(self, dp: float = 1.2, min_dist: int = 50,
                 param1: int = 50, param2: int = 30,
                 min_radius: int = 20, max_radius: int = 50):
        """
        Initialize the CircleDetector with Hough Circle parameters.
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        logger.debug(f"Initialized CircleDetector with dp={dp}, min_dist={min_dist}, param1={param1}, param2={param2}, min_radius={min_radius}, max_radius={max_radius}")

    def detect_circles(self, image_path: str, image_id: str) -> List[Dict]:
        """
        Detect circles in the image and return their properties.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Image file '{image_path}' not found.")
                raise FileNotFoundError(f"Image file '{image_path}' not found.")
            logger.info(f"Processing image for circle detection: {image_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            detected_circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.dp,
                minDist=self.min_dist,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )

            circles = []

            if detected_circles is not None:
                detected_circles = np.round(detected_circles[0, :]).astype("int")
                logger.info(f"Detected {len(detected_circles)} circles in image ID={image_id}")
                for (x, y, r) in detected_circles:
                    circle_id = str(uuid.uuid4())
                    bounding_box = (x - r, y - r, 2 * r, 2 * r)
                    circle = {
                        "id": circle_id,
                        "image_id": image_id,
                        "centroid_x": int(x),  # Cast to Python int
                        "centroid_y": int(y),  # Cast to Python int
                        "radius": int(r),      # Cast to Python int
                        "bounding_box_x": int(bounding_box[0]),  # Cast to Python int
                        "bounding_box_y": int(bounding_box[1]),  # Cast to Python int
                        "bounding_box_width": int(bounding_box[2]),  # Cast to Python int
                        "bounding_box_height": int(bounding_box[3])  # Cast to Python int
                    }
                    circles.append(circle)
                    logger.debug(f"Detected circle: ID={circle_id}, Centroid=({x},{y}), Radius={r}")
            else:
                logger.warning(f"No circles detected in image ID={image_id}.")

            return circles
        except Exception as e:
            logger.error(f"Error during circle detection: {e}")
            raise
