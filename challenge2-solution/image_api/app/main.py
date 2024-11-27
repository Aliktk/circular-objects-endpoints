# app/main.py

from fastapi import FastAPI, HTTPException, Query
from typing import List
from .database import initialize_db, get_db_connection
from .schemas import ImageResponse
from .utils import process_and_store_csv
import base64
from .logger import setup_logging
import logging
import os

app = FastAPI(title="Image Processing API")

# Initialize logging
setup_logging()

# Initialize the database
initialize_db()

# Process and store CSV data on startup if database is empty
@app.on_event("startup")
def startup_event():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM images")
    count = cursor.fetchone()[0]
    conn.close()
    if count == 0:
        csv_path = os.path.join("data", "images.csv")
        if os.path.exists(csv_path):
            process_and_store_csv(csv_path)
        else:
            logging.error(f"CSV file not found at {csv_path}")
    else:
        logging.info("Database already initialized with image data.")

@app.get("/images/", response_model=List[ImageResponse])
def get_images(depth_min: float = Query(..., description="Minimum depth"),
               depth_max: float = Query(..., description="Maximum depth")):
    logging.info(f"GET /images/ called with depth_min={depth_min}, depth_max={depth_max}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT depth, image FROM images WHERE depth BETWEEN ? AND ?", (depth_min, depth_max))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        logging.error(f"No images found between depths {depth_min} and {depth_max}")
        raise HTTPException(status_code=404, detail="No images found in the specified depth range.")
    
    image_responses = []
    for row in rows:
        try:
            encoded_image = base64.b64encode(row['image']).decode('utf-8')
            image_responses.append(ImageResponse(depth=row['depth'], image=encoded_image))
        except Exception as e:
            logging.error(f"Error encoding image at depth {row['depth']}: {e}")
    
    return image_responses

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API. Visit /docs for API documentation."}
