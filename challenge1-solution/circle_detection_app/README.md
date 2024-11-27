# Contour & Watershed Detection

## Overview

The **Contour & Watershed Detection** method employs classical computer vision techniques to identify and analyze coins within an image. This approach is effective for detecting circular objects and handling overlapping coins through segmentation.

## Features

- **Contour Detection:** Identifies coin boundaries based on their shapes.
- **Watershed Algorithm:** Segments overlapping coins for accurate individual detection.
- **Radius & Area Calculation:** Estimates the size of each detected coin.

## How It Works

1. **Contour Detection:**
   - Converts the input image to grayscale.
   - Applies Gaussian Blur to reduce noise.
   - Utilizes the Canny edge detector to identify edges.
   - Detects circles using the Hough Circle Transform based on adjustable parameters.

2. **Watershed Algorithm:**
   - Converts the image to grayscale and applies thresholding.
   - Removes noise using morphological operations.
   - Identifies sure background and foreground areas.
   - Applies the Watershed algorithm to segment overlapping coins.

## Usage Instructions

1. **Select Detection Method:**
   - Open the application.
   - Choose **"Contour Detection"** or **"Watershed Algorithm"** from the detection methods.

2. **Adjust Parameters:**
   - **Contour Detection Parameters:**
     - `dp`: Inverse ratio of the accumulator resolution to the image resolution.
     - `minDist`: Minimum distance between detected centers.
     - `param1`: Higher threshold for the Canny edge detector.
     - `param2`: Accumulator threshold for circle centers.
     - `minRadius`: Minimum circle radius.
     - `maxRadius`: Maximum circle radius.
   - **Watershed Parameters:**
     - `Markers`: Number of markers for the watershed algorithm.

3. **Process Image:**
   - Click the **"Process"** button.
   - View the processed image with detected coins and segmentation overlays.
   - Segmented masks for each coin will appear in the **"Segmented Masks"** section.

4. **Save Results:**
   - After processing, click **"Save Result"** to export the annotated image.

## Parameters Explained

### Contour Detection

- **dp:** Controls the resolution of the accumulator. Higher values may detect fewer circles.
- **minDist:** Ensures detected circles are adequately spaced to avoid multiple detections of the same coin.
- **param1 & param2:** Thresholds for edge detection and center detection respectively.
- **minRadius & maxRadius:** Define the size range of coins to detect.

### Watershed Algorithm

- **Markers:** Determines the number of initial regions to segment. Higher values can handle more overlapping coins.

## Notes

- **Image Quality:** High-resolution images with clear coin boundaries yield better detection results.
- **Overlapping Coins:** The Watershed method is particularly effective in accurately segmenting overlapping coins.
- **Performance:** Classical methods are computationally efficient and suitable for real-time applications on standard hardware.

---

## Endpoints

### 1. Upload Image and Detect Circles

- **URL:** `/upload_image`
- **Method:** `POST`
- **Description:** Upload an image to detect circles (coins) using contour-based detection and optionally evaluate the model against ground truth data.
- **Request:**
  - **Form Data:**
    - `image`: Image file (supported formats: PNG, JPG, JPEG, BMP)
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "image_id": "unique-image-id",
      "filename": "uploaded_image.jpg",
      "detected_circles": [
        {
          "id": "circle-id-1",
          "centroid": { "x": 150, "y": 200 },
          "radius": 30
        },
        // ... more circles
      ]
    }
    ```

### 2. Retrieve All Circles for an Image

- **URL:** `/images/{image_id}/circles`
- **Method:** `GET`
- **Description:** Retrieve all detected circles for a specific image.
- **Path Parameters:**
  - `image_id` (string): Unique identifier of the image.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    [
      {
        "id": "circle-id-1",
        "centroid": { "x": 150, "y": 200 },
        "radius": 30
      },
      // ... more circles
    ]
    ```

### 3. Retrieve Details of a Specific Circle

- **URL:** `/circles/{circle_id}`
- **Method:** `GET`
- **Description:** Retrieve properties of a specific circle.
- **Path Parameters:**
  - `circle_id` (string): Unique identifier of the circle.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "id": "circle-id-1",
      "centroid": { "x": 150, "y": 200 },
      "radius": 30
    }
    ```

### 4. Download Image with Overlayed Masks

- **URL:** `/download_image_with_mask/{image_id}`
- **Method:** `GET`
- **Description:** Download the original image with detected circles overlayed. Optionally include ground truth circles via a CSV file.
- **Path Parameters:**
  - `image_id` (string): Unique identifier of the image.
- **Query Parameters:**
  - `ground_truth_file` (file, optional): CSV file with columns `centroid_x`, `centroid_y`, `radius`.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:** Image file with overlays.

### 5. Evaluate Model Against Ground Truth

- **URL:** `/evaluate_model`
- **Method:** `POST`
- **Description:** Evaluate the circle detection model's performance against ground truth data provided in a CSV file.
- **Request:**
  - **Form Data:**
    - `ground_truth_file`: CSV file with columns `image_id`, `centroid_x`, `centroid_y`, `radius`.
- **Response:**
  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "Precision": 0.95,
      "Recall": 0.90,
      "F1_Score": 0.925
    }
    ```

## Usage Instructions

1. **Upload an Image:**
   - Send a `POST` request to `/upload_image` with the image file.
   - Receive `image_id` and details of detected circles.

2. **Retrieve Detected Circles:**
   - Use `GET /images/{image_id}/circles` to fetch all circles for the uploaded image.

3. **Get Specific Circle Details:**
   - Use `GET /circles/{circle_id}` to obtain properties of a particular circle.

4. **Download Annotated Image:**
   - Use `GET /download_image_with_mask/{image_id}` to download the image with overlays.
   - Optionally include a ground truth CSV file for comparison.

5. **Evaluate Model Performance:**
   - Use `POST /evaluate_model` with a ground truth CSV file to receive evaluation metrics.

## Notes

- **Supported Formats:** Ensure images are in supported formats (PNG, JPG, JPEG, BMP).
- **Ground Truth CSV:** Must contain columns `image_id`, `centroid_x`, `centroid_y`, `radius` for accurate evaluation.
- **Error Handling:** The API returns appropriate HTTP status codes and error messages for invalid requests.

---