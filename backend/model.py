from ultralytics import YOLO
import os

MODEL_PATH = os.path.join("..","runs","detect","train-8","weights","best.pt")

model = YOLO(MODEL_PATH)

def detect(img):
    results = model(img)[0]
    detections = []

    if results.boxes:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "type": model.names[cls],
                "confidence": round(conf, 3)
            })

    return detections