# app/database.py

import sqlite3
from typing import List, Dict, Optional
import os
from .logger import logger

DATABASE_NAME = "circle_detection.db"

def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to SQLite database: {DATABASE_NAME}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def create_tables():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        logger.debug("Creating tables if they do not exist.")

        # Create images table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT UNIQUE NOT NULL
        )
        """)
        logger.info("Ensured 'images' table exists.")

        # Create circles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS circles (
            id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            centroid_x INTEGER NOT NULL,
            centroid_y INTEGER NOT NULL,
            radius INTEGER NOT NULL,
            bounding_box_x INTEGER NOT NULL,
            bounding_box_y INTEGER NOT NULL,
            bounding_box_width INTEGER NOT NULL,
            bounding_box_height INTEGER NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
        )
        """)
        logger.info("Ensured 'circles' table exists.")

        conn.commit()
        logger.debug("Database tables created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed after creating tables.")

def insert_image(image_id: str, filename: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO images (id, filename) VALUES (?, ?)
        """, (image_id, filename))
        conn.commit()
        logger.info(f"Inserted image: ID={image_id}, Filename={filename}")
    except sqlite3.IntegrityError as e:
        logger.error(f"Integrity error inserting image: {e}")
        raise
    except sqlite3.Error as e:
        logger.error(f"Error inserting image: {e}")
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed after inserting image.")

def insert_circle(circle: Dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO circles (id, image_id, centroid_x, centroid_y, radius, bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            circle['id'],
            circle['image_id'],
            circle['centroid_x'],
            circle['centroid_y'],
            circle['radius'],
            circle['bounding_box_x'],
            circle['bounding_box_y'],
            circle['bounding_box_width'],
            circle['bounding_box_height']
        ))
        conn.commit()
        logger.info(f"Inserted circle: ID={circle['id']} for Image ID={circle['image_id']}")
    except sqlite3.IntegrityError as e:
        logger.error(f"Integrity error inserting circle: {e}")
        raise
    except sqlite3.Error as e:
        logger.error(f"Error inserting circle: {e}")
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed after inserting circle.")

def get_circles_by_image(image_id: str) -> List[Dict]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, centroid_x, centroid_y, radius, bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height
        FROM circles
        WHERE image_id = ?
        """, (image_id,))
        rows = cursor.fetchall()
        conn.close()
        circles = []
        for row in rows:
            circle = {
                "id": row["id"],
                "centroid_x": int(row["centroid_x"]),
                "centroid_y": int(row["centroid_y"]),
                "radius": int(row["radius"]),
                "bounding_box_x": int(row["bounding_box_x"]),
                "bounding_box_y": int(row["bounding_box_y"]),
                "bounding_box_width": int(row["bounding_box_width"]),
                "bounding_box_height": int(row["bounding_box_height"])
            }
            circles.append(circle)
        logger.debug(f"Retrieved {len(circles)} circles for Image ID={image_id}")
        return circles
    except sqlite3.Error as e:
        logger.error(f"Error retrieving circles for Image ID={image_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after retrieving circles.")

def get_circle_by_id(circle_id: str) -> Optional[Dict]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, centroid_x, centroid_y, radius, bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height
        FROM circles
        WHERE id = ?
        """, (circle_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            circle = {
                "id": row["id"],
                "centroid_x": int(row["centroid_x"]),
                "centroid_y": int(row["centroid_y"]),
                "radius": int(row["radius"]),
                "bounding_box_x": int(row["bounding_box_x"]),
                "bounding_box_y": int(row["bounding_box_y"]),
                "bounding_box_width": int(row["bounding_box_width"]),
                "bounding_box_height": int(row["bounding_box_height"])
            }
            logger.debug(f"Retrieved circle: ID={circle_id}")
            return circle
        logger.warning(f"Circle ID={circle_id} not found.")
        return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving circle ID={circle_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after retrieving circle.")

def get_image_filename(image_id: str) -> Optional[str]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT filename FROM images WHERE id = ?
        """, (image_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            filename = row["filename"]
            logger.debug(f"Retrieved filename for Image ID={image_id}: {filename}")
            return filename
        logger.warning(f"Image ID={image_id} not found.")
        return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving filename for Image ID={image_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after retrieving image filename.")

def get_all_images() -> List[Dict]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, filename FROM images
        """)
        rows = cursor.fetchall()
        conn.close()
        images = []
        for row in rows:
            image = {
                "id": row["id"],
                "filename": row["filename"]
            }
            images.append(image)
        logger.debug(f"Retrieved {len(images)} images from database.")
        return images
    except sqlite3.Error as e:
        logger.error(f"Error retrieving all images: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after retrieving all images.")
