# app/database.py

import sqlite3
from sqlite3 import Connection
from typing import List, Tuple
import os

DATABASE_NAME = "images.db"

def get_db_connection() -> Connection:
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    if not os.path.exists(DATABASE_NAME):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                depth REAL UNIQUE,
                image BLOB
            )
        """)
        conn.commit()
        conn.close()
