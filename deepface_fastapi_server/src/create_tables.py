#!/usr/bin/env python3

from database import create_db_and_tables

if __name__ == "__main__":
    create_db_and_tables()
    print("Database tables created successfully!")
    print("New sent_plates table is ready for tracking sent plates.") 