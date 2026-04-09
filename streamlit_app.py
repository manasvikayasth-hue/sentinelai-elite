import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import sqlite3
import base64

# ================= CONFIG =================
st.set_page_config(page_title="SentinelAI Elite", layout="wide")

st.title("🛡️ SentinelAI Elite")
st.markdown("### AI Surveillance + Face Recognition System")

# ================= ALERT SOUND =================
def play_alert():
    audio_file = open("alert.wav", "rb")
    audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# ================= DATABASE =================
conn = sqlite3.connect("logs.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    person_name TEXT,
    threat TEXT,
    image_path TEXT
)
""")
conn.commit()

def log_detection(name, threat, path):
    c.execute(
        "INSERT INTO logs (timestamp, person_name, threat, image_path) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, threat, path)
    )
    conn.commit()

# ================= FACE DATABASE =================
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = []
    st.session_state.known_names = []

# ================= FACE UPLOAD =================
st.sidebar.header("👤 Add Known Faces")

uploaded = st.sidebar.file_uploader("Upload Face", type=["jpg","png","jpeg"])

if uploaded:
    name = st.sidebar.text_input("Enter Name")

    if st.sidebar.button("Save Face"):
        img = face_recognition.load_image_file(uploaded)
        enc = face_recognition.face_encodings(img)

        if enc:
            st.session_state.known_encodings.append(enc[0])
            st.session_state.known_names.append(name)
            st.sidebar.success(f"{name} added!")

# ================= CAMERA =================
run = st.checkbox("Start Monitoring")

frame_window = st.empty()
alert_box = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not accessible")

    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 Resize for speed
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # 🔥 Process every 4 frames
        if frame_count % 4 == 0:
            faces = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, faces)

            for (top, right, bottom, left), face_encoding in zip(faces, encodings):

                top*=2; right*=2; bottom*=2; left*=2

                name = "Unknown"
                if len(st.session_state.known_encodings) > 0:
                    matches = face_recognition.compare_faces(st.session_state.known_encodings, face_encoding)
                    distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoding)

                    if len(distances) > 0:
                        best = np.argmin(distances)
                        if matches[best]:
                            name = st.session_state.known_names[best]

                threat = "SAFE" if name!="Unknown" else "THREAT"
                color = (0,255,0) if threat=="SAFE" else (0,0,255)

                cv2.rectangle(frame,(left,top),(right,bottom),color,2)
                cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

                # 🚨 ALERT
                if threat == "THREAT":
                    if not os.path.exists("detections"):
                        os.makedirs("detections")

                    filename = f"detections/{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)

                    log_detection(name, threat, filename)

                    alert_box.error("🚨 UNKNOWN PERSON DETECTED!")
                    play_alert()

        frame_window.image(frame, channels="BGR")

    cap.release()

# ================= INCIDENT FEED =================
st.subheader("📜 Incident Feed")

rows = c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5").fetchall()

for r in rows:
    st.write(f"{r[1]} | {r[2]} | {r[3]}")
    if os.path.exists(r[4]):
        st.image(r[4], width=200)