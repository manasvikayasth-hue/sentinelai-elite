import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import sqlite3
from datetime import datetime
import os

st.set_page_config(page_title="SentinelAI Elite", layout="centered")

st.title("🛡️ SentinelAI Elite")
st.write("Real-Time AI Surveillance (Lightweight)")

# Load YOLO model
model = YOLO("yolov8n.pt")

# DB setup
conn = sqlite3.connect("logs.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    label TEXT,
    image_path TEXT
)
""")
conn.commit()

# Create detections folder
os.makedirs("detections", exist_ok=True)

# Camera input (WORKS ON CLOUD)
st.subheader("📷 Camera Feed")
camera_image = st.camera_input("Take a picture")

if camera_image:
    image = np.array(Image.open(camera_image))

    results = model(image)

    detected_labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"detections/{timestamp}.jpg"
    Image.fromarray(image).save(path)

    # Save to DB
    for label in detected_labels:
        c.execute("INSERT INTO logs (time, label, image_path) VALUES (?, ?, ?)",
                  (timestamp, label, path))
    conn.commit()

    st.image(image, caption="Detected Image")

    if detected_labels:
        st.success(f"Detected: {', '.join(detected_labels)}")
    else:
        st.warning("No objects detected")

# Dashboard
st.subheader("📜 Incident Dashboard")

rows = c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 10").fetchall()

for row in rows:
    st.write(f"🕒 {row[1]} | {row[2]}")
    if os.path.exists(row[3]):
        st.image(row[3], width=250)