import sqlite3
import json
import os

print("DB path at runtime:", os.path.abspath("videoDB.sqlite"))

def add_video(id, name, video):
    print("➡️ add_video called with id=", id, "frames=", len(video))
    conn = sqlite3.connect("videoDB.sqlite")
    cursor = conn.cursor()

    video_json = json.dumps(video)

    cursor.execute(
        "INSERT INTO users (user_id, name, video) VALUES (?, ?, ?)",
        (id, name, video_json)
    )

    conn.commit()
    conn.close()

def update(id, name, video):
    print("🔄 update called with id=", id, "frames=", len(video))
    conn = sqlite3.connect("videoDB.sqlite", timeout=5)
    cursor = conn.cursor()

    video_json = json.dumps(video)

    cursor.execute(
        "UPDATE users SET name = ?, video = ? WHERE user_id = ?",
        (name, video_json, id)
    )

    conn.commit()
    conn.close()

def both(id, name, video):
    conn = sqlite3.connect("videoDB.sqlite")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE user_id = ?", (id,))
    exists = cursor.fetchone() is not None
    if(exists):
        update(id, name, video)
    else:
        add_video(id, name, video)

    conn.commit()
    conn.close()