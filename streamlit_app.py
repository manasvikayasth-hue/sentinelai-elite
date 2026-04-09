import streamlit as st
import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import sqlite3
import time
import threading
import face_recognition
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="SentinelAI ELITE", layout="wide")

DETECTIONS_DIR = "detections"
KNOWN_DIR = "known_faces"
DB_PATH = "database.db"

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    person TEXT,
    status TEXT,
    image_path TEXT
)
""")
conn.commit()

def log_event(person, status, path):
    c.execute("INSERT INTO logs (timestamp, person, status, image_path) VALUES (?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), person, status, path))
    conn.commit()

# ---------------- EMAIL ---------------- #
def send_email(image_path, label):
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    to = os.environ.get("EMAIL_TO")

    if not user or not password or not to:
        return

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = f"🚨 SentinelAI Alert: {label}"

    part = MIMEBase('application', 'octet-stream')
    with open(image_path, 'rb') as f:
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
    msg.attach(part)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(user, password)
        server.send_message(msg)
        server.quit()
    except:
        pass

# ---------------- SOUND ---------------- #
def play_sound():
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ---------------- LOAD FACES ---------------- #
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_DIR):
    img_path = os.path.join(KNOWN_DIR, file)
    img = face_recognition.load_image_file(img_path)
    enc = face_recognition.face_encodings(img)
    if enc:
        known_encodings.append(enc[0])
        known_names.append(os.path.splitext(file)[0])

# ---------------- MODEL ---------------- #
model = YOLO("yolov8n.pt")

# ---------------- LOGIN ---------------- #
if "user" not in st.session_state:
    st.session_state.user = None

def login():
    st.title("🔐 SentinelAI Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in ["admin", "viewer"] and p == "1234":
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.user:
    login()
    st.stop()

# ---------------- UI ---------------- #
st.title("🛰️ SentinelAI ELITE")

run = st.toggle("Start Monitoring")
FRAME = st.empty()

# ---------------- MONITOR ---------------- #
if run:
    cap = cv2.VideoCapture(0)
    last_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):

            matches = face_recognition.compare_faces(known_encodings, face_enc)
            name = "Unknown"

            if True in matches:
                name = known_names[matches.index(True)]

            # draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # 🚨 ALERT ONLY FOR UNKNOWN
            if name == "Unknown" and time.time() - last_alert > 5:
                st.warning("🚨 UNKNOWN PERSON DETECTED")

                threading.Thread(target=play_sound).start()

                filename = f"{DETECTIONS_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)

                log_event(name, "INTRUDER", filename)

                threading.Thread(target=send_email, args=(filename, "Unknown Person")).start()

                last_alert = time.time()

        FRAME.image(frame, channels="BGR")

    cap.release()

# ---------------- INCIDENT FEED ---------------- #
st.subheader("📋 Incident Feed")

c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5")
rows = c.fetchall()

for r in rows:
    st.write(f"🚨 {r[2]} | {r[3]} | {r[1]}")
    if os.path.exists(r[4]):
        st.image(r[4])