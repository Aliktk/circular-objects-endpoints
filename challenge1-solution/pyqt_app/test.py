import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import uuid
import os
import csv
import pandas as pd  # Optional, for enhanced CSV handling

# Define a dataclass for circle properties
from dataclasses import dataclass

@dataclass
class Circle:
    id: str
    centroid: Tuple[int, int]
    radius: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)

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
        self.circles: List[Circle] = []

    def detect_circles(self, image_path: str) -> None:
        """
        Detect circles in the image and assign unique IDs.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

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

        self.circles = []  # Reset circles list

        if detected_circles is not None:
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                circle_id = str(uuid.uuid4())  # Assign a unique ID
                bounding_box = (x - r, y - r, 2 * r, 2 * r)
                circle = Circle(
                    id=circle_id,
                    centroid=(x, y),
                    radius=r,
                    bounding_box=bounding_box
                )
                self.circles.append(circle)
        else:
            print("No circles detected.")

    def get_all_circles(self) -> List[Dict]:
        """
        Retrieve a list of all detected circles with their IDs and bounding boxes.
        """
        return [{
            'id': circle.id,
            'bounding_box': circle.bounding_box
        } for circle in self.circles]

    def get_circle_properties(self, circle_id: str) -> Optional[Dict]:
        """
        Given a circle ID, return its bounding box, centroid, and radius.
        """
        for circle in self.circles:
            if circle.id == circle_id:
                return {
                    'bounding_box': circle.bounding_box,
                    'centroid': circle.centroid,
                    'radius': circle.radius
                }
        print(f"No circle found with ID: {circle_id}")
        return None

    def display_results(self, image_path: str, semantic: bool = False, ground_truth: Optional[List[Dict]] = None) -> None:
        """
        Display the detected circles and optionally the semantic segmentation.
        Optionally overlay ground truth circles for comparison.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        overlay_img = img.copy()
        semantic_img = np.zeros_like(img)

        for circle in self.circles:
            x, y = circle.centroid
            r = circle.radius
            # Draw the outer circle
            cv2.circle(overlay_img, (x, y), r, (0, 255, 0), 2)  # Detected circles in green
            # Draw the center of the circle
            cv2.circle(overlay_img, (x, y), 2, (0, 0, 255), 3)  # Center in red
            # Create mask for semantic segmentation
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            masked = cv2.bitwise_and(img, img, mask=mask)
            semantic_img = cv2.add(semantic_img, masked)

        if ground_truth:
            for gt_circle in ground_truth:
                x, y = gt_circle['centroid']
                r = gt_circle['radius']
                # Draw ground truth circles in blue
                cv2.circle(overlay_img, (x, y), r, (255, 0, 0), 2)
                cv2.circle(overlay_img, (x, y), 2, (0, 0, 255), 3)

        if semantic:
            # Convert BGR to RGB for displaying using matplotlib
            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            semantic_rgb = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(overlay_rgb)
            title = "Detected Circles (Green)"
            if ground_truth:
                title += " and Ground Truth (Blue)"
            plt.title(title)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(semantic_rgb)
            plt.title("Semantic Segmentation")
            plt.axis('off')

            plt.show()
        else:
            # Display only the overlay image
            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay_rgb)
            plt.title("Detected Circles")
            plt.axis('off')
            plt.show()

    def evaluate_model(self, ground_truth: List[Dict], tolerance: int = 10) -> Dict[str, float]:
        """
        Evaluate the circle detection model against ground truth data.

        Parameters:
            ground_truth (List[Dict]): List of ground truth circles with 'centroid' and 'radius'.
            tolerance (int): Maximum distance and radius difference to consider a detection as True Positive.

        Returns:
            Dict[str, float]: Evaluation metrics including Precision, Recall, and F1-Score.
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        gt_matched = set()
        det_matched = set()

        for gt_idx, gt_circle in enumerate(ground_truth):
            gt_centroid = gt_circle['centroid']
            gt_radius = gt_circle['radius']
            for det_idx, det_circle in enumerate(self.circles):
                det_centroid = det_circle.centroid
                det_radius = det_circle.radius
                distance = np.linalg.norm(np.array(gt_centroid) - np.array(det_centroid))
                radius_diff = abs(gt_radius - det_radius)
                if distance <= tolerance and radius_diff <= tolerance:
                    if gt_idx not in gt_matched and det_idx not in det_matched:
                        true_positives += 1
                        gt_matched.add(gt_idx)
                        det_matched.add(det_idx)

        false_positives = len(self.circles) - len(det_matched)
        false_negatives = len(ground_truth) - len(gt_matched)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }

    def save_circles_to_csv(self, csv_path: str) -> None:
        """
        Save all detected circles to a CSV file.

        Parameters:
            csv_path (str): Path to the CSV file where data will be saved.
        """
        if not self.circles:
            print("No circles to save.")
            return

        # Define CSV headers
        headers = ['id', 'centroid_x', 'centroid_y', 'radius',
                   'bounding_box_x', 'bounding_box_y',
                   'bounding_box_width', 'bounding_box_height']

        # Prepare data rows
        rows = []
        for circle in self.circles:
            row = {
                'id': circle.id,
                'centroid_x': circle.centroid[0],
                'centroid_y': circle.centroid[1],
                'radius': circle.radius,
                'bounding_box_x': circle.bounding_box[0],
                'bounding_box_y': circle.bounding_box[1],
                'bounding_box_width': circle.bounding_box[2],
                'bounding_box_height': circle.bounding_box[3]
            }
            rows.append(row)

        # Write to CSV using csv module
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Detected circles saved to '{csv_path}'.")

    def load_circles_from_csv(self, csv_path: str) -> None:
        """
        Load circles from a CSV file and populate the circles attribute.

        Parameters:
            csv_path (str): Path to the CSV file from which data will be loaded.
        """
        if not os.path.exists(csv_path):
            print(f"CSV file '{csv_path}' does not exist.")
            return

        # Read CSV using csv module
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            self.circles = []  # Reset existing circles
            for row in reader:
                try:
                    circle = Circle(
                        id=row['id'],
                        centroid=(int(row['centroid_x']), int(row['centroid_y'])),
                        radius=int(row['radius']),
                        bounding_box=(
                            int(row['bounding_box_x']),
                            int(row['bounding_box_y']),
                            int(row['bounding_box_width']),
                            int(row['bounding_box_height'])
                        )
                    )
                    self.circles.append(circle)
                except (ValueError, KeyError) as e:
                    print(f"Error parsing row {row}: {e}")

        print(f"Loaded {len(self.circles)} circles from '{csv_path}'.")


# Example Usage
if __name__ == "__main__":
    # Initialize the detector with desired parameters
    detector = CircleDetector(dp=1.2, min_dist=50, param1=50, param2=30, min_radius=20, max_radius=50)
    
    # Path to the image
    image_path = "test.jpg"  # Replace with your image path
    
    # Detect circles
    detector.detect_circles(image_path)
    
    # Retrieve all circles
    all_circles = detector.get_all_circles()
    print("All Detected Circles:")
    for circle_info in all_circles:
        print(circle_info)
    
    # Get properties of a specific circle (example using first circle)
    if all_circles:
        first_circle_id = all_circles[0]['id']
        properties = detector.get_circle_properties(first_circle_id)
        print(f"\nProperties of Circle ID {first_circle_id}:")
        print(properties)
    
    # Save detected circles to CSV
    csv_path = "detected_circles.csv"  # Specify your desired CSV file path
    detector.save_circles_to_csv(csv_path)
    
    # Display results with semantic segmentation
    detector.display_results(image_path, semantic=True)
    
    # Evaluation (Check if ground truth file exists)
    gt_path = "ground_truth.txt"  # Replace with your ground truth file path if different
    if os.path.exists(gt_path):
        ground_truth = load_ground_truth(gt_path)
        evaluation_metrics = detector.evaluate_model(ground_truth, tolerance=10)
        print("\nModel Evaluation Metrics:")
        print(evaluation_metrics)
        
        # Optionally, display detected and ground truth circles together
        detector.display_results(image_path, semantic=True, ground_truth=ground_truth)
    else:
        print(f"\nGround truth file '{gt_path}' not found. Skipping evaluation.")
    
    # Load circles from CSV (demonstration)
    print("\nLoading circles from CSV...")
    detector_loaded = CircleDetector()
    detector_loaded.load_circles_from_csv(csv_path)
    loaded_circles = detector_loaded.get_all_circles()
    print("Loaded Circles from CSV:")
    for circle_info in loaded_circles:
        print(circle_info)
