import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import os
from datetime import datetime

st.set_page_config(page_title="SentinelAI Lite", layout="centered")

st.title("🛡️ SentinelAI Lite")
st.write("Simple AI Face Recognition System")

KNOWN_FACES_DIR = "known_faces"
DETECTIONS_DIR = "detections"

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{file}")
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(file.split(".")[0])

st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    results = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "UNKNOWN"

        if True in matches:
            name = known_names[matches.index(True)]

        results.append(name)

    st.subheader("🔍 Results")

    if results:
        for name in results:
            if name == "UNKNOWN":
                st.error("⚠️ UNKNOWN PERSON DETECTED")
            else:
                st.success(f"✅ SAFE: {name}")
    else:
        st.warning("No face detected")

    # Save detection image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{DETECTIONS_DIR}/{timestamp}.jpg"
    Image.fromarray(image).save(save_path)

    st.image(image, caption="Processed Image")