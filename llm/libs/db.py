import sqlite3
import json
from datetime import datetime

def init_db():
    conn = get_db_connection()
    
    # Create the raw_descriptions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_description TEXT NOT NULL,
            image_url TEXT NOT NULL,
            code TEXT NOT NULL,
            app_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create the processed_descriptions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS processed_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            processed_description TEXT NOT NULL,
            code TEXT NOT NULL,
            app_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect('llm_descriptions.db')

def create_raw_description(raw_description, image_url, code, app_type, status):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO raw_descriptions (raw_description, image_url, code, app_type, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (raw_description, image_url, code, app_type, status))
        conn.commit()

        return cursor.lastrowid
    finally:
        conn.close()
        
def create_processed_description(processed_description, code, app_type, status):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO processed_descriptions (processed_description, code, app_type, status)
            VALUES (?, ?, ?, ?)
        ''', (processed_description, code, app_type, status))
        conn.commit()

        return cursor.lastrowid
    finally:
        conn.close()

def update_raw_description_status_and_code(id: int, raw_description: str, status: str, code: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE raw_descriptions SET status = ?, code = ?, raw_description = ? WHERE id = ?
        ''', (status, code, raw_description, id))
        conn.commit()
    finally:
        conn.close()
def update_raw_description_status(id: int, status: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE raw_descriptions SET status = ? WHERE id = ?
        ''', (status, id))
        conn.commit()
    finally:
        conn.close()

def update_processed_description_status_and_code(id: int, status: str, code: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE processed_descriptions SET status = ?, code = ? WHERE id = ?
        ''', (status, code, id))
        conn.commit()
    finally:
        conn.close()


def get_raw_descriptions_by_status(status: str) -> list[tuple]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM raw_descriptions WHERE status = ?
        ''', (status,))
        return cursor.fetchall()
    finally:
        conn.close()

def get_raw_description_by_id(id: int) -> list[tuple]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM raw_descriptions WHERE id = ?
        ''', (id,))
        return cursor.fetchall()
    finally:
        conn.close()

def get_raw_descriptions_by_code(code: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM raw_descriptions WHERE code = ?
        ''', (code,))
        return cursor.fetchall()
    finally:
        conn.close()
        
def get_raw_description_by_code(code: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM raw_descriptions WHERE code = ?
        ''', (code,))
        return cursor.fetchone()
    finally:
        conn.close() 
    
def get_processed_description_by_id(id: int) -> list[tuple]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM processed_descriptions WHERE id = ?
        ''', (id,))
        return cursor.fetchall()
    finally:
        conn.close()
        
def get_processed_descriptions_by_code(code: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM processed_descriptions WHERE code = ?
        ''', (code,))
        return cursor.fetchall()
    finally:
        conn.close()

def get_raw_descriptions_paginated(limit: int, offset: int, code: str | None = None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        base_query = "SELECT * FROM raw_descriptions"
        params = []
        if code:
            base_query += " WHERE code = ?"
            params.append(code)
        
        base_query += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(base_query, tuple(params))
        items = cursor.fetchall()
        
        # Get total count for pagination metadata
        count_query = "SELECT COUNT(*) FROM raw_descriptions"
        count_params = []
        if code:
            count_query += " WHERE code = ?"
            count_params.append(code)
        cursor.execute(count_query, tuple(count_params))
        total_count = cursor.fetchone()[0]
        
        return items, total_count
    finally:
        conn.close()

def get_processed_descriptions_paginated(limit: int, offset: int, code: str | None = None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        base_query = "SELECT * FROM processed_descriptions"
        params = []
        if code:
            base_query += " WHERE code = ?"
            params.append(code)

        base_query += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(base_query, tuple(params))
        items = cursor.fetchall()

        # Get total count for pagination metadata
        count_query = "SELECT COUNT(*) FROM processed_descriptions"
        count_params = []
        if code:
            count_query += " WHERE code = ?"
            count_params.append(code)
        cursor.execute(count_query, tuple(count_params))
        total_count = cursor.fetchone()[0]

        return items, total_count
    finally:
        conn.close()

