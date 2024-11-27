# app/database.py

import sqlite3
from sqlite3 import Connection
import os
from .logger import logger

# Get the absolute path of the current directory (app/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the absolute path to the database file
DATABASE_NAME = os.path.join(CURRENT_DIR, "yellow_detection.db")

def get_db_connection() -> Connection:
    logger.debug(f"Connecting to database at {DATABASE_NAME}")
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    if not os.path.exists(DATABASE_NAME):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create circular_objects table
        cursor.execute("""
            CREATE TABLE circular_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                bounding_box TEXT NOT NULL, -- Stored as JSON string
                centroid TEXT NOT NULL,     -- Stored as JSON string
                radius REAL NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Initialized the database and created tables.")
    else:
        logger.info("Database already exists. Skipping initialization.")

# Initialize the database on module import
initialize_database()
