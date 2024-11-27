# YOLOv8 Segmentation

## Overview

The **YOLOv8 Segmentation** method enhances object detection by providing precise pixel-level segmentation of coins within images. This advanced technique allows for detailed analysis, including calculating the radius and area of each detected coin.

## Features

- **Pixel-Level Precision:** Generates accurate segmentation masks for each detected coin.
- **Integrated Detection & Segmentation:** Combines object detection with segmentation for comprehensive analysis.
- **Adjustable Thresholds:** Customize confidence and NMS thresholds to balance precision and recall.

## How It Works

1. **Model Loading:**
   - The application initializes the YOLOv8 segmentation model (`yolov8_segmentation.pt`) during startup.

2. **Inference:**
   - Upon processing, the model analyzes the input image to detect and segment coins.
   - Outputs segmentation masks along with bounding boxes and confidence scores.

3. **Post-Processing:**
   - Draws bounding boxes, centroids, and overlays segmentation masks on the processed image.
   - Applies Non-Maximum Suppression (NMS) to refine detections.

4. **Analysis:**
   - Calculates the radius and area of each segmented coin based on the mask.

## Usage Instructions

1. **Select Detection Method:**
   - Open the application.
   - Choose **"YOLOv8 Segmentation"** from the detection methods.

2. **Adjust Parameters:**
   - **Confidence Threshold:** Minimum confidence level to consider a segmentation valid.
   - **NMS Threshold:** Threshold for Non-Maximum Suppression to filter overlapping segments.

3. **Process Image:**
   - Click the **"Process"** button.
   - View the processed image with segmentation masks and analytical overlays indicating coin properties.

4. **Save Results:**
   - After processing, click **"Save Result"** to export the annotated image.

## Parameters Explained

- **Confidence Threshold:**
  - **Range:** 10% to 100%
  - **Description:** Filters out segmentations with confidence below the set value to ensure accuracy.

- **NMS Threshold:**
  - **Range:** 10% to 100%
  - **Description:** Determines the overlap threshold for suppressing multiple segmentations of the same coin.

## Notes

- **Model Files:** Ensure `yolov8_segmentation.pt` is correctly placed and the path is specified in the application.
- **Class Names:** The model should be trained to recognize and segment the class `'coin'`. Adjust class filters in the code if necessary.
- **Performance:** YOLOv8 Segmentation provides detailed analysis but may require more computational resources compared to detection-only methods.


---
## Endpoints

### 1. Upload Image for YOLOv8 Segmentation

- **URL:** `/segment-image`
- **Method:** `POST`
- **Description:** Upload an image to perform segmentation using the YOLOv8 segmentation model.
- **Request:**
  - **Form Data:**
    - `file`: Image file (supported formats: PNG, JPG, JPEG, BMP)
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "image_id": 3,
      "filename": "segmented_coins.jpg",
      "segmentation_objects": [
        {
          "id": 1,
          "segmentation_mask": {
            "mask_data": [
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 1.0, 0.0, 0.0],
              // ... more mask data
            ]
          },
          "centroid": { "x": 189.79, "y": 312.43 },
          "radius": 38.24,
          "area": 4542.02
        },
        // ... more segmentation objects
      ]
    }
    ```

### 2. Retrieve Segmentation Results for an Image

- **URL:** `/images/{image_id}/segmentation`
- **Method:** `GET`
- **Description:** Retrieve segmentation results for a specific image, including detailed information about each segmented coin.
- **Path Parameters:**
  - `image_id` (integer): Unique identifier of the image.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "image_id": 3,
      "segmentation_objects": [
        {
          "id": 1,
          "segmentation_mask": {
            "mask_data": [
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 1.0, 0.0, 0.0],
              // ... more mask data
            ]
          },
          "centroid": { "x": 189.79, "y": 312.43 },
          "radius": 38.24,
          "area": 4542.02
        },
        // ... more segmentation objects
      ]
    }
    ```

### 3. Retrieve Radius of a Specific Segmented Object

- **URL:** `/circular-objects/{object_id}/radius`
- **Method:** `GET`
- **Description:** Retrieve the radius of a specific segmented circular object detected by YOLOv8.
- **Path Parameters:**
  - `object_id` (integer): Unique identifier of the segmented circular object.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "id": 1,
      "radius": 38.24
    }
    ```

## Usage Instructions

1. **Upload an Image for Segmentation:**
   - Send a `POST` request to `/segment-image` with the image file.
   - Receive `image_id` and details of segmented objects.

2. **Retrieve Segmentation Results:**
   - Use `GET /images/{image_id}/segmentation` to fetch all segmentation objects for the uploaded image.

3. **Get Radius of a Specific Segmented Object:**
   - Use `GET /circular-objects/{object_id}/radius` to obtain the radius of a particular segmented object.

## Parameters Explained

- **Segmentation Mask:**
  - `mask_data`: A 2D array representing the binary mask of the segmented object.

- **Centroid:**
  - `x`, `y`: Coordinates representing the center of the segmented object.

- **Radius:**
  - Estimated radius of the segmented object based on the mask.

- **Area:**
  - Calculated area of the segmented object in pixel units.

## Notes

- **Model Integration:** Ensure the YOLOv8 segmentation model (`yolov8_segmentation.pt`) is correctly loaded and configured in the application.
- **Supported Formats:** Images must be in supported formats (PNG, JPG, JPEG, BMP) for accurate segmentation.
- **Performance:** YOLOv8 Segmentation provides detailed analysis but may require more computational resources compared to detection-only methods.
- **Error Handling:** The API returns appropriate HTTP status codes and error messages for invalid requests or processing errors.

## Example Usage

### Upload Image for Segmentation

```bash
curl -X POST "http://localhost:8000/segment-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"