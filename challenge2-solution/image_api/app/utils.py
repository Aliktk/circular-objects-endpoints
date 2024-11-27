import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import base64
from .database import get_db_connection
from .models import ImageData
import logging

def resize_image(image_array: np.ndarray, new_width: int = 150) -> np.ndarray:
    height, width = image_array.shape
    if width == 0 or height == 0:
        raise ValueError("Image has invalid dimensions for resizing.")
    aspect_ratio = height / width
    new_height = max(int(new_width * aspect_ratio), 1)  # Ensure new_height is at least 1
    resized_image = cv2.resize(image_array, (new_width, new_height))
    return resized_image

def apply_color_map(image_array: np.ndarray) -> np.ndarray:
    # Apply a custom color map (e.g., COLORMAP_JET)
    colored_image = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)
    return colored_image

def process_and_store_csv(csv_path: str):
    logging.info(f"Starting CSV processing for {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return
    
    # Verify that 'depth' column exists
    if 'depth' not in df.columns:
        logging.error("CSV does not contain 'depth' column.")
        return
    
    # Expected number of pixel columns
    expected_pixel_columns = 200
    pixel_columns = [col for col in df.columns if col.startswith('col')]
    
    if len(pixel_columns) != expected_pixel_columns:
        logging.error(f"CSV should contain exactly {expected_pixel_columns} pixel columns, found {len(pixel_columns)}.")
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()

    for index, row in df.iterrows():
        try:
            depth = float(row['depth'])
        except ValueError:
            logging.error(f"Row {index}: Invalid depth value '{row['depth']}'. Skipping row.")
            continue

        # Extract pixel values
        pixel_values = row[pixel_columns].values

        # Handle missing or non-numeric values
        if pd.isnull(pixel_values).any():
            logging.error(f"Row {index}: Contains NaN values in pixel data. Skipping row.")
            continue

        try:
            # Ensure all pixel values are integers between 0 and 255
            pixel_values = pixel_values.astype(int)
            if not np.all((0 <= pixel_values) & (pixel_values <= 255)):
                logging.error(f"Row {index}: Pixel values out of range [0, 255]. Skipping row.")
                continue
            pixel_values = pixel_values.astype(np.uint8)
        except Exception as e:
            logging.error(f"Row {index}: Error converting pixel values to uint8: {e}. Skipping row.")
            continue

        # Check if the number of pixel values is exactly 200
        if len(pixel_values) != expected_pixel_columns:
            logging.error(f"Row {index}: Expected {expected_pixel_columns} pixel values, got {len(pixel_values)}. Skipping row.")
            continue

        # Calculate image dimensions
        image_width = 200
        image_height = len(pixel_values) // image_width

        if image_height == 0:
            logging.error(f"Row {index}: Computed image height is 0. Skipping row.")
            continue

        try:
            # Reshape pixel values to 2D array
            image_array = pixel_values.reshape((image_height, image_width))
        except Exception as e:
            logging.error(f"Row {index}: Error reshaping pixel values: {e}. Skipping row.")
            continue

        try:
            # Resize image
            resized_image = resize_image(image_array, new_width=150)
        except Exception as e:
            logging.error(f"Row {index}: Error resizing image: {e}. Skipping row.")
            continue

        try:
            # Apply color map
            colored_image = apply_color_map(resized_image)
        except Exception as e:
            logging.error(f"Row {index}: Error applying color map: {e}. Skipping row.")
            continue

        try:
            # Convert image to bytes
            is_success, buffer = cv2.imencode(".png", colored_image)
            if not is_success:
                logging.error(f"Row {index}: Failed to encode image at depth {depth}. Skipping row.")
                continue
            image_bytes = buffer.tobytes()
        except Exception as e:
            logging.error(f"Row {index}: Error encoding image: {e}. Skipping row.")
            continue

        try:
            # Insert into DB
            cursor.execute("INSERT OR IGNORE INTO images (depth, image) VALUES (?, ?)", (depth, image_bytes))
            logging.info(f"Row {index}: Processed and stored image at depth {depth}")
        except Exception as e:
            logging.error(f"Row {index}: Database insertion error: {e}. Skipping row.")
            continue

    conn.commit()
    conn.close()
    logging.info("CSV processing completed.")
