# Coin Detector Application

Welcome to the **Coin Detector Application**, a PyQt-based desktop tool designed to detect and analyze coins in images using multiple computer vision techniques. This application offers four distinct detection methods to cater to various user needs:

1. **Contour Detection**
2. **Watershed Algorithm**
3. **YOLOv8 Object Detection**
4. **YOLOv8 Segmentation**

## Table of Contents

- [Coin Detector Application](#coin-detector-application)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
    - [1. Contour Detection](#1-contour-detection)
    - [2. Watershed Algorithm](#2-watershed-algorithm)
    - [3. YOLOv8 Object Detection](#3-yolov8-object-detection)
    - [4. YOLOv8 Segmentation](#4-yolov8-segmentation)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Download YOLOv8 Models](#4-download-yolov8-models)
    - [5. Run the Application](#5-run-the-application)
  - [Usage](#usage)
    - [Running the Application](#running-the-application)
    - [Detection Methods](#detection-methods)
      - [1. Contour Detection](#1-contour-detection-1)
      - [2. Watershed Algorithm](#2-watershed-algorithm-1)
      - [3. YOLOv8 Detection](#3-yolov8-detection)
      - [4. YOLOv8 Segmentation](#4-yolov8-segmentation-1)
  - [Dependencies](#dependencies)
  - [Project Structure](#project-structure)
  - [License](#license)
  - [Contact](#contact)

## Features

### 1. Contour Detection

Utilizes classical computer vision techniques to identify coin boundaries based on their contours. Ideal for simple and quick detections without the need for complex models.

### 2. Watershed Algorithm

Employs the watershed segmentation technique to separate overlapping coins and accurately delineate each coin's boundaries, enhancing detection accuracy in cluttered images.

### 3. YOLOv8 Object Detection

Integrates the state-of-the-art YOLOv8 model for real-time object detection. This method provides high accuracy and speed, making it suitable for applications requiring precise localization of coins.

### 4. YOLOv8 Segmentation

Leverages YOLOv8's segmentation capabilities to not only detect coins but also generate precise segmentation masks. This allows for detailed analysis, including calculating the radius and area of each detected coin.

## Installation

Follow these steps to set up the **Coin Detector Application** on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/coin-detector-app.git
cd coin-detector-app
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

* **Windows:**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download YOLOv8 Models

Ensure you have the YOLOv8 detection and segmentation models (`yolov8_detection.pt` and `yolov8_segmentation.pt`) download from [here](http//:google.com) placed in the project directory or specify their paths in the code.

### 5. Run the Application

```python
python coin_detector_app.py
```
## Usage

### Running the Application

1. **Launch the Application:**
   * Run the Python script to open the Coin Detector GUI.
2. **Load an Image:**
   * Click the **"Load Image"** button and select an image containing coins.
3. **Select Detection Method:**
   * Choose one of the four detection methods:
     * **Contour Detection**
     * **Watershed Algorithm**
     * **YOLOv8 Detection**
     * **YOLOv8 Segmentation**
4. **Adjust Parameters:**
   * Depending on the selected method, adjust the available parameters using the sliders provided.
5. **Process the Image:**
   * Click the **"Process"** button to execute the selected detection method.
   * The processed image with overlays will appear in the **"Processed Image with Overlay"** section.
   * Segmented masks will be displayed in the **"Segmented Masks"** section.
6. **Save the Result:**
   * After processing, click the **"Save Result"** button to save the processed image.

### Detection Methods

#### 1. Contour Detection

* **Parameters:**
  * `dp`: Inverse ratio of the accumulator resolution to the image resolution.
  * `minDist`: Minimum distance between detected centers.
  * `param1`: Higher threshold for the Canny edge detector.
  * `param2`: Accumulator threshold for circle centers.
  * `minRadius`: Minimum circle radius.
  * `maxRadius`: Maximum circle radius.
* **Usage:**
  * Adjust the sliders as needed.
  * Click **"Process"** to detect coins based on contours.

#### 2. Watershed Algorithm

* **Parameters:**
  * `Markers`: Number of markers for the watershed algorithm.
* **Usage:**
  * Set the desired number of markers.
  * Click **"Process"** to segment overlapping coins.

#### 3. YOLOv8 Detection

* **Parameters:**
  * `Confidence Threshold`: Minimum confidence to consider a detection valid.
  * `NMS Threshold`: Non-Maximum Suppression threshold to eliminate overlapping boxes.
* **Usage:**
  * Adjust the confidence and NMS thresholds.
  * Click **"Process"** to perform object detection using YOLOv8.

#### 4. YOLOv8 Segmentation

* **Parameters:**
  * `Confidence Threshold`: Minimum confidence to consider a segmentation valid.
  * `NMS Threshold`: Non-Maximum Suppression threshold to eliminate overlapping segments.
* **Usage:**
  * Adjust the confidence and NMS thresholds.
  * Click **"Process"** to perform segmentation using YOLOv8.

## Dependencies

The application relies on the following Python packages:

* **PyQt5:** For building the graphical user interface.
* **OpenCV (`cv2`):** For image processing and computer vision tasks.
* **Ultralytics YOLOv8 (`ultralytics`):** For object detection and segmentation models.
* **NumPy:** For numerical operations.
* **Logging:** For tracking application events and errors.

Install all dependencies using the provided `requirements.txt` file.

## Project Structure
```bash

coin-detector-app/
├── coin_detector_app.py
├── requirements.txt
├── yolov8_detection.pt
├── yolov8_segmentation.pt
├── README.md
├── app.log
└── assets/
    └── sample_image.jpg
```
* **coin_detector_app.py:** Main application script.
* **requirements.txt:** Lists all Python dependencies.
* **yolov8_detection.pt:** Pre-trained YOLOv8 detection model.
* **yolov8_segmentation.pt:** Pre-trained YOLOv8 segmentation model.
* **README.md:** Project documentation.
* **app.log:** Log file capturing application events and errors.
* **assets/:** Directory containing sample images and other assets.

## License

This project is licensed under the [MIT License]().

## Contact

For any questions or support, please contact:

* **Email:** [nawazktk99@gmail.com]()
* **GitHub:** [Aliktk](https://github.com/Aliktk)
