import cv2
import numpy as np
from ultralytics import YOLO

# Load model once
model = YOLO("yolov8n.pt")


def process_detection(file):
    try:
        # Convert file to image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run detection
        results = model(image)

        # Draw boxes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                label = f"{conf:.2f}"

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return image

    except Exception as e:
        return None