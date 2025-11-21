import sqlite3

conn = sqlite3.connect("videoDB.sqlite")
cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
               user_id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT,
               video TEXT
)
""")
conn.commit()