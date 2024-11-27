# Coin Detection and Segmentation API

Welcome to the **Coin Detection and Segmentation API** project! This application provides robust endpoints for detecting and analyzing coins in images using three distinct methodologies: Contour-Based Detection, YOLOv8 Object Detection, and Object Segmentation. Whether you're interested in simple contour detection or advanced object segmentation with area and radius calculations, this API has you covered.

## Table of Contents

- [Coin Detection and Segmentation API](#coin-detection-and-segmentation-api)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
    - [1. Contour-Based Detection](#1-contour-based-detection)
    - [2. YOLOv8 Object Detection](#2-yolov8-object-detection)
    - [3. Object Segmentation](#3-object-segmentation)
  - [Technologies Used](#technologies-used)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Download and Place the YOLOv8 Model](#4-download-and-place-the-yolov8-model)
    - [5. Initialize the Database](#5-initialize-the-database)
  - [Usage](#usage)
    - [Running the API](#running-the-api)

## Overview

The **Coin Detection and Segmentation API** is designed to process images containing coins and provide detailed information about each detected coin, including its location, radius, and area. The application leverages advanced computer vision techniques and machine learning models to ensure accurate and efficient detection and analysis.

## Features

### 1. Contour-Based Detection

**Description:**
Contour-based detection is a classical computer vision technique that identifies object boundaries by detecting continuous curves or shapes in an image. This method is effective for detecting coins based on their circular shapes.

**How It Works:**

* Converts the input image to grayscale.
* Applies Gaussian Blur to reduce noise.
* Utilizes edge detection (Canny) to identify edges.
* Finds contours from the detected edges.
* Filters contours based on shape and size to identify coins.
* Calculates the centroid, radius, and area for each detected coin.

**Endpoint:**

* **URL:** `/detect-contours`
* **Method:** `POST`
* **Description:** Upload an image to detect coins using contour-based detection.

### 2. YOLOv8 Object Detection

**Description:**
YOLOv8 (You Only Look Once version 8) is a state-of-the-art object detection model known for its speed and accuracy. By training YOLOv8 on a custom dataset of coins, the model can effectively identify and localize coins within images.

**How It Works:**

* **Training Phase:**
  * Collect and annotate a custom dataset containing various coin images.
  * Train the YOLOv8 model on this dataset to recognize coins.
* **Inference Phase:**
  * The trained YOLOv8 model processes the input image.
  * Detects coins with bounding boxes and confidence scores.
  * Extracts relevant information such as centroid, radius, and area.

**Endpoint:**

* **URL:** `/detect-yolov8`
* **Method:** `POST`
* **Description:** Upload an image to detect coins using the YOLOv8 object detection model.

### 3. Object Segmentation

**Description:**
Object segmentation goes beyond detection by delineating the exact pixels that constitute each object—in this case, each coin. This method allows for precise calculations of the area and radius of each segmented coin.

**How It Works:**

* Receives an input image containing coins.
* Utilizes a segmentation model (e.g., YOLOv8 with segmentation capabilities) to identify and segment each coin.
* For each segmented coin:
  * Calculates the centroid coordinates.
  * Estimates the radius assuming a circular shape.
  * Computes the area based on the segmented mask.

**Endpoint:**

* **URL:** `/segment-image`
* **Method:** `POST`
* **Description:** Upload an image to perform object segmentation, extracting detailed information about each coin.

## Technologies Used

* **Python 3.8+**
* **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python.
* **Uvicorn:** A lightning-fast ASGI server implementation, using uvloop and httptools.
* **YOLOv8:** Advanced object detection and segmentation model.
* **OpenCV:** Open Source Computer Vision Library for image processing.
* **SQLite:** Lightweight disk-based database.
* **Pydantic:** Data validation and settings management using Python type annotations.
* **Ultralytics:** Provides the YOLOv8 implementation.
* **Logging:** Comprehensive logging for monitoring and debugging.

## Project Structure

```bash
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── logger.py
│   ├── utils.py
│   └── yolo_model/
│       └── best.pt
├── logs/
│   ├── app_info.log
│   └── app_error.log
├── storage/
│   ├── uploads/
│   └── masks/
├── yolo_segmentation.db
├── requirements.txt
├── README.md
└── assets/
    └── coin_detection.png

```

* **app/** : Contains all application-related modules and code.
* **main.py** : Defines API endpoints and application logic.
* **database.py** : Manages database connections and schema initialization.
* **models.py** : Defines Pydantic models for data validation.
* **logger.py** : Configures logging for the application.
* **utils.py** : Contains utility functions and model inference logic.
* **yolo_model/** : Stores the pre-trained YOLOv8 model file (`best.pt`).
* **logs/** : Stores log files capturing application events and errors.
* **storage/** :
* **uploads/** : Directory to store uploaded images.
* **masks/** : (Optional) Directory to store segmentation masks.
* **yolo_segmentation.db** : SQLite database file storing image and segmentation data.
* **requirements.txt** : Lists all Python dependencies.
* **README.md** : Project documentation.
* **assets/** : Contains images and other static assets used in the README.

## Installation

Follow these steps to set up the **Coin Detection and Segmentation API** on your local machine.

```
git clone https://github.com/Aliktk/circular-objects-detection.git
cd coin-detection-segmentation-api
```

### 1. Clone the Repository

cd coin-detection-segmentation-api
2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

* **Windows:**

`venv\Scripts\activate`

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download and Place the YOLOv8 Model

Ensure you have the YOLOv8 segmentation model (`best.pt`) placed in the `app/yolo_model/` directory.
Find Model [All Models Here]([http//:google.com](https://drive.google.com/drive/folders/1VzoVqrBbmqfAP9vKWFymhaatcWGtMllt?usp=sharing))

### 5. Initialize the Database

The application will automatically initialize the SQLite database (`yolo_segmentation.db`) upon first run. Ensure that the `app/` directory has the necessary write permissions.

## Usage

### Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```


Once the server is running, access the API documentation at [http://127.0.0.1:8000/docs]() to interact with the endpoints via Swagger UI.
