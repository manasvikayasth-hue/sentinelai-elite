from flask import Blueprint, request, send_file
from app.services.detect_service import process_detection
import cv2

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("file")

    if not file:
        return {"error": "No file uploaded"}, 400

    image = process_detection(file)

    if image is None:
        return {"error": "Processing failed"}, 500

    # Convert image to bytes
    _, buffer = cv2.imencode('.jpg', image)

    return send_file(
        buffer.tobytes(),
        mimetype='image/jpeg'
    )
