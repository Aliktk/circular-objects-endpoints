# YOLOv8 Object Detection

## Overview

The **YOLOv8 Object Detection** method integrates the state-of-the-art YOLOv8 model to accurately detect and localize coins within images. This approach offers high precision and speed, making it ideal for applications requiring real-time detection.

## Features

- **High Accuracy:** Leverages YOLOv8's advanced architecture for precise coin detection.
- **Real-Time Processing:** Optimized for quick inference suitable for dynamic environments.
- **Adjustable Thresholds:** Fine-tune confidence and NMS thresholds for optimal results.

## How It Works

1. **Model Loading:**
   - The application initializes the YOLOv8 detection model (`yolov8_detection.pt`) during startup.

2. **Inference:**
   - Upon processing, the model analyzes the input image to detect coins.
   - Outputs bounding boxes with confidence scores for each detected coin.

3. **Post-Processing:**
   - Draws bounding boxes and centroids on the processed image.
   - Applies Non-Maximum Suppression (NMS) to eliminate redundant detections.

## Usage Instructions

1. **Select Detection Method:**
   - Open the application.
   - Choose **"YOLOv8 Detection"** from the detection methods.

2. **Adjust Parameters:**
   - **Confidence Threshold:** Minimum confidence level to consider a detection valid.
   - **NMS Threshold:** Threshold for Non-Maximum Suppression to filter overlapping boxes.

3. **Process Image:**
   - Click the **"Process"** button.
   - View the processed image with bounding boxes and centroids indicating detected coins.

4. **Save Results:**
   - After processing, click **"Save Result"** to export the annotated image.

## Parameters Explained

- **Confidence Threshold:**
  - **Range:** 10% to 100%
  - **Description:** Filters out detections with confidence below the set value to reduce false positives.

- **NMS Threshold:**
  - **Range:** 10% to 100%
  - **Description:** Determines the overlap threshold for suppressing multiple detections of the same coin.

## Notes

- **Model Files:** Ensure `yolov8_detection.pt` is correctly placed and the path is specified in the application.
- **Class Names:** The model should be trained to recognize the class `'coin'`. Adjust class filters in the code if necessary.
- **Performance:** YOLOv8 offers rapid detection speeds, but performance may vary based on hardware capabilities.

---
## Endpoints

### 1. Upload Image for YOLOv8 Detection

- **URL:** `/upload-image`
- **Method:** `POST`
- **Description:** Upload an image to detect coins using the YOLOv8 object detection model.
- **Request:**
  - **Form Data:**
    - `file`: Image file (supported formats: PNG, JPG, JPEG, BMP)
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "image_id": 1,
      "filename": "coins.jpg",
      "detections": [
        {
          "id": 1,
          "bounding_box": { "x1": 100.5, "y1": 150.75, "x2": 200.25, "y2": 250.5 },
          "confidence": 0.98,
          "class": "coin"
        },
        // ... more detections
      ]
    }
    ```

### 2. Retrieve Circular Objects for an Image

- **URL:** `/images/{image_id}/circular-objects`
- **Method:** `GET`
- **Description:** Retrieve all detected circular objects (coins) for a specific image.
- **Path Parameters:**
  - `image_id` (integer): Unique identifier of the image.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "image_id": 1,
      "circular_objects": [
        {
          "id": 1,
          "bounding_box": { "x1": 100.5, "y1": 150.75, "x2": 200.25, "y2": 250.5 }
        },
        // ... more objects
      ]
    }
    ```

### 3. Retrieve Radius of a Specific Circular Object

- **URL:** `/circular-objects/{object_id}/radius`
- **Method:** `GET`
- **Description:** Retrieve the radius of a specific circular object detected by YOLOv8.
- **Path Parameters:**
  - `object_id` (integer): Unique identifier of the circular object.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "id": 1,
      "radius": 30.5
    }
    ```

## Usage Instructions

1. **Upload an Image for Detection:**
   - Send a `POST` request to `/upload-image` with the image file.
   - Receive `image_id` and details of detected coins.

2. **Retrieve Detected Circular Objects:**
   - Use `GET /images/{image_id}/circular-objects` to fetch all circular objects for the uploaded image.

3. **Get Radius of a Specific Object:**
   - Use `GET /circular-objects/{object_id}/radius` to obtain the radius of a particular circular object.

## Parameters Explained

- **Bounding Box:**
  - `x1`, `y1`: Coordinates of the top-left corner.
  - `x2`, `y2`: Coordinates of the bottom-right corner.
  
- **Confidence:**
  - Represents the model's confidence in the detection (range: 0.0 to 1.0).

- **Class:**
  - The detected object's class label (e.g., "coin").

## Notes

- **Model Integration:** Ensure the YOLOv8 detection model (`yolov8_detection.pt`) is correctly loaded and configured in the application.
- **Supported Formats:** Images must be in supported formats (PNG, JPG, JPEG, BMP) for accurate detection.
- **Performance:** YOLOv8 provides rapid detection speeds; however, hardware capabilities may affect processing times.
- **Error Handling:** The API returns appropriate HTTP status codes and error messages for invalid requests or processing errors.

---
