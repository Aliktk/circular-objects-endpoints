from PIL import Image
import io
import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import os
import sqlite3
def fetch_images(db_path: str, limit: int = 16, depth_min: float = None, depth_max: float = None):
    """
    Fetch images from the SQLite database.

    :param db_path: Path to the SQLite database file.
    :param limit: Maximum number of images to fetch.
    :param depth_min: Minimum depth value for filtering.
    :param depth_max: Maximum depth value for filtering.
    :return: List of tuples containing depth and image bytes.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Construct the SQL query with optional depth filtering
        query = "SELECT depth, image FROM images"
        params = []

        if depth_min is not None and depth_max is not None:
            query += " WHERE depth BETWEEN ? AND ?"
            params.extend([depth_min, depth_max])

        query += " ORDER BY depth ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("No images found with the specified criteria.")
            return []

        return rows

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []

def decode_image(image_data: bytes):
    """
    Decode binary image data to a PIL Image.

    :param image_data: Binary image data.
    :return: PIL.Image object.
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        # Convert image to RGB if it's in a different mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def display_image_grid(images, grid_size=(4, 4)):
    """
    Display images in a grid using matplotlib.

    :param images: List of tuples containing depth and PIL.Image objects.
    :param grid_size: Tuple indicating the grid dimensions (rows, cols).
    """
    rows, cols = grid_size
    total_slots = rows * cols
    actual_images = len(images)

    if actual_images == 0:
        print("No images to display.")
        return

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle('Image Grid', fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx in range(total_slots):
        ax = axes[idx]
        if idx < actual_images:
            depth, img = images[idx]
            if img is not None:
                ax.imshow(img)
                ax.set_title(f"Depth: {depth}")
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Invalid Image', horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
        else:
            # Hide any unused subplots
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """
    Main function to execute the image retrieval and display.
    """
    # Set up argument parser
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images.db")
    parser = argparse.ArgumentParser(description="Display a grid of images from SQLite database.")
    parser.add_argument('--db-path', type=str, default=DB_PATH, help="Path to the SQLite database file.")
    parser.add_argument('--limit', type=int, default=16, help="Number of images to display (default: 16).")
    parser.add_argument('--depth-min', type=float, default=None, help="Minimum depth value for filtering.")
    parser.add_argument('--depth-max', type=float, default=None, help="Maximum depth value for filtering.")
    parser.add_argument('--grid-rows', type=int, default=4, help="Number of rows in the image grid (default: 4).")
    parser.add_argument('--grid-cols', type=int, default=4, help="Number of columns in the image grid (default: 4).")
    args = parser.parse_args()

    # Fetch images from the database
    fetched_rows = fetch_images(args.db_path, limit=args.limit, depth_min=args.depth_min, depth_max=args.depth_max)

    if not fetched_rows:
        sys.exit(0)

    # Decode images
    images = []
    for idx, (depth, image_data) in enumerate(fetched_rows):
        img = decode_image(image_data)
        if img is not None:
            images.append((depth, img))
        else:
            print(f"Row {idx}: Failed to decode image.")

    # Display images in a grid
    grid_size = (args.grid_rows, args.grid_cols)
    display_image_grid(images, grid_size=grid_size)

if __name__ == "__main__":
    main()
