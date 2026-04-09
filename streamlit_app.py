import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import sqlite3
import os
import time
import hashlib
import pandas as pd

# ---------- CONFIG ----------
st.set_page_config(page_title="SentinelAI Elite", layout="wide")

# ---------- MODEL ----------
model = YOLO("yolov8n.pt")

# ---------- DB ----------
conn = sqlite3.connect("logs.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    time TEXT,
    label TEXT,
    image_path TEXT
)
""")

conn.commit()
os.makedirs("detections", exist_ok=True)

# ---------- HASH ----------
def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ---------- CREATE USERS ----------
def create_users():
    users = [
        ("mahek","1234","admin"),
        ("mahi","1234","user"),
        ("khushbu","1234","user"),
        ("nancy","1234","user"),
        ("taksh","1234","user"),
        ("urvi","1234","user"),
    ]
    for u in users:
        try:
            c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                      (u[0], hash_pass(u[1]), u[2]))
        except:
            pass
    conn.commit()

create_users()

# ---------- SESSION ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "last_saved" not in st.session_state:
    st.session_state.last_saved = 0

# ---------- LOGIN ----------
def login():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        res = c.execute("SELECT * FROM users WHERE username=? AND password=?",
                        (u, hash_pass(p))).fetchone()

        if res:
            st.session_state.user = res[1]
            st.session_state.role = res[3]
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------- LOGOUT ----------
def logout():
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# ---------- AUTH ----------
if not st.session_state.user:
    login()
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.title(f"👤 {st.session_state.user}")
st.sidebar.write(f"Role: {st.session_state.role}")

if st.sidebar.button("🚪 Logout"):
    logout()

menu = ["Live", "Logs", "Analytics"]
if st.session_state.role == "admin":
    menu.append("Admin")

choice = st.sidebar.radio("Menu", menu)

# ---------- LIVE ----------
if choice == "Live":
    st.title("🎥 Live Surveillance")

    run = st.toggle("Start Camera")

    frame_window = st.empty()
    status = st.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)

        labels = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                labels.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if labels:
            status.warning(f"Detected: {', '.join(set(labels))}")

        if labels and time.time() - st.session_state.last_saved > 3:
            st.session_state.last_saved = time.time()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"detections/{ts}.jpg"
            cv2.imwrite(path, frame)

            for l in set(labels):
                c.execute("INSERT INTO logs VALUES (NULL, ?, ?, ?, ?)",
                          (st.session_state.user, ts, l, path))
            conn.commit()

        frame_window.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()

# ---------- LOGS ----------
elif choice == "Logs":
    st.title("📜 Logs")

    rows = c.execute("SELECT * FROM logs WHERE user=? ORDER BY id DESC",
                     (st.session_state.user,)).fetchall()

    data = []

    for r in rows:
        data.append({"time": r[2], "label": r[3]})
        st.write(f"{r[2]} | {r[3]}")
        if os.path.exists(r[4]):
            st.image(r[4], width=300)

    # 🔥 ALWAYS SHOW EXPORT BUTTON
    df = pd.DataFrame(data)

    st.download_button(
        "📤 Export CSV",
        df.to_csv(index=False),
        file_name="logs.csv"
    )

# ---------- ANALYTICS ----------
elif choice == "Analytics":
    st.title("📊 Analytics")

    rows = c.execute("SELECT label FROM logs WHERE user=?",
                     (st.session_state.user,)).fetchall()

    if rows:
        df = pd.DataFrame([r[0] for r in rows], columns=["Object"])
        st.bar_chart(df["Object"].value_counts())

# ---------- ADMIN ----------
elif choice == "Admin":
    st.title("👑 Admin Panel")

    users = c.execute("SELECT * FROM users").fetchall()
    for u in users:
        st.write(f"{u[1]} ({u[3]})")

    rows = c.execute("SELECT label FROM logs").fetchall()

    if rows:
        df = pd.DataFrame([r[0] for r in rows], columns=["Object"])
        st.bar_chart(df["Object"].value_counts())