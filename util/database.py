"""
By Jaerong
Create & import database for the project
"""
import sqlite3 as lite


def create_database(database_path: str):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        cur.execute("drop table if exists deafening")
    conn.close()
